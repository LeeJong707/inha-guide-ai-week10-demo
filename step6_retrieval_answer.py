"""
6단계: 학사 문서 하이브리드 검색 및 답변

특징:
- OpenAI API를 사용하지 않는다.
- 직접 학습한 질문 의도 분류 모델을 사용한다.
- 모델 신뢰도가 낮으면 기존 규칙 기반 판단으로 fallback한다.
- 실제 교과목명이 질문에 포함되면 curriculum 안전망을 사용한다.
- 단, 로드맵/트랙 문맥에서는 교과목명 안전망이 가로채지 않는다.
- 현재 지원하지 않는 학사 영역은 out_of_scope로 처리한다.
- 교육과정 질문은 구조화 메타데이터 검색을 사용한다.
- 졸업요건 / 로드맵 질문은 로컬 임베딩 검색을 사용한다.
- 근거 문서 정보를 함께 표시한다.

실행:
python step6_retrieval_answer.py
"""

import re
import os
from pathlib import Path
from typing import Any
from functools import lru_cache
from expansion_search import explicit_extension_intent, answer_extension
from question_policy import input_policy
from request_context import get_request_context, request_text

import chromadb
import joblib
from sentence_transformers import SentenceTransformer


# =========================================================
# 기본 설정
# =========================================================

MODEL_NAME = os.environ.get("INHA_E5_MODEL", "intfloat/multilingual-e5-small")

CHROMA_DB_PATH = "chroma_db_v3"
COLLECTION_NAME = "inha_academic_documents_v3"

INTENT_MODEL_PATH = Path(
    "models/intent_classifier_v2.joblib"
)

INTENT_CONFIDENCE_THRESHOLD = 0.50

SEARCH_RESULT_COUNT = 6
MIN_FINAL_SCORE = 0.50

DEFAULT_PROGRAM_YEARS = 3


INTENT_TO_DOCUMENT_TYPE = {
    "graduation": "졸업요건",
    "curriculum": "교육과정",
    "roadmap": "로드맵",
    "substitute_course": "대체과목",
    "certificate_credit": "자격증학점인정",
}


OUT_OF_SCOPE_MESSAGE = "현재는 학사정보 검색만 지원합니다. 해당 정보는 현재 자료에 포함되어 있지 않습니다."


def out_of_scope_answer(question: str) -> str:
    normalized = normalize_text(question)
    subject = next((word for word in UNSUPPORTED_SCOPE_KEYWORDS if word in normalized), None)
    if subject:
        return f"현재 보유한 자료에는 {subject} 정보가 포함되어 있지 않습니다."
    return "현재는 학사정보 검색만 지원합니다."


UNSUPPORTED_SCOPE_KEYWORDS = [
    "장학금",
    "등록금",
    "수강신청",
    "학사일정",
    "학사 일정",
    "중간고사",
    "기말고사",
    "시험 일정",
    "시험일정",
    "휴학",
    "복학",
    "자퇴",
    "성적 정정",
    "성적정정",
    "교수님 이메일",
    "교수 이메일",
    "셔틀버스",
    "도서관",
    "취업지원",
    "취업 지원",
]


# =========================================================
# 텍스트 정규화
# =========================================================


def normalize_text(
    text: str,
) -> str:
    """
    비교를 위해 공백과 특수문자를 단순화한다.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^\w가-힣·]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =========================================================
# 졸업요건 답변 추출
# =========================================================


def extract_academic_credit_answer(question: str, document_text: str) -> str | None:
    """Extract numbers only from the selected admission section and program length."""
    q = normalize_text(question)
    text = normalize_text(document_text)
    # Cohort exceptions and advanced programs need their full original context.
    if re.search(r"\d{4}\s*학번", q) or "전공심화" in q:
        return None
    year_match = re.search(r"([23])\s*년제", q)
    years = int(year_match.group(1)) if year_match else DEFAULT_PROGRAM_YEARS
    transfer = "편입" in q
    admission = "편입학" if transfer else "신입학"
    if transfer and not document_text.lstrip().startswith("편입학"):
        return None
    if not transfer and "신입학" not in document_text:
        return None
    section = text.split("전공심화")[0]
    total = re.search(rf"{years}년제\s*(\d+)\s*학점", section)
    category = re.search(rf"{years}년제\s*교양\s*(\d+)\s*학점\s*전공\s*(\d+)\s*학점", section)
    transfer_major = re.search(rf"{years}년제\s*전공\s*(\d+)\s*학점", section)
    asks_major, asks_liberal = "전공" in q, "교양" in q
    asks_total = any(word in q for word in ["총", "전체", "졸업학점", "졸업 이수학점"])
    if not asks_major and not asks_liberal:
        asks_total = True
    lines = []
    if asks_total and total:
        lines.append(f"{years}년제 졸업학점은 {total.group(1)}학점입니다.")
    if transfer:
        if asks_major and transfer_major:
            lines.append(f"편입학 이후 전공 최소취득학점은 {transfer_major.group(1)}학점입니다.")
    elif category:
        if asks_liberal:
            lines.append(f"교양은 최소 {category.group(1)}학점입니다.")
        if asks_major:
            lines.append(f"전공은 최소 {category.group(2)}학점입니다.")
    if not lines:
        return None
    return "\n".join(lines) + f"\n({admission} {years}년제 기준)"


# =========================================================
# 로드맵 답변 추출
# =========================================================


def extract_roadmap_answer(
    question: str,
    document_text: str,
) -> str | None:
    """
    로드맵 문서에서 질문에 맞는 정보를 추출한다.

    지원 예:
    - 3학년 2학기 로드맵 과목
    - 데이터분석은 로드맵에서 몇 학년에 나오는지
    - 생성형 AI 프로그래밍이 포함된 트랙
    - 로드맵에 없는 특정 과목 확인
    """

    normalized_question = normalize_text(question).replace("에이전트", "agent")
    requested_tracks = re.findall(r"([A-Za-z0-9가-힣·]+)\s*트랙", question)
    supported = {"ai", "bigdata", "빅데이터", "ai·bigdata", "인공지능", "학과", "전체", "해당", "어떤", "어느", "무슨"}
    if any(name.lower() not in supported for name in requested_tracks):
        return "현재 로드맵 데이터는 AI·BigData 트랙만 지원합니다. 요청한 다른 트랙은 현재 자료에서 확인되지 않습니다."


    compact_question = (
        normalized_question
        .replace(
            " ",
            "",
        )
    )

    # -----------------------------------------------------
    # 로드맵 트랙명 추출
    # -----------------------------------------------------

    track_match = re.search(
        r"^\s*([^\n]+?트랙)\s*로드맵",
        document_text,
        flags=re.MULTILINE,
    )

    if track_match:
        track_name = (
            track_match
            .group(1)
            .strip()
        )
    else:
        track_name = "해당 트랙"

    # -----------------------------------------------------
    # 학년 / 학기 / 과목 파싱
    # -----------------------------------------------------

    roadmap_entries = []

    entry_pattern = re.compile(
        r"([1-4])\s*학년\s*"
        r"([1-2])\s*학기\s*:"
        r"\s*([^\n]+)"
    )

    for match in entry_pattern.finditer(
        document_text
    ):
        grade = match.group(1)
        semester = match.group(2)

        course_text = (
            match
            .group(3)
            .strip()
        )

        courses = [
            course.strip()
            for course
            in course_text.split(",")
            if course.strip()
        ]

        roadmap_entries.append(
            {
                "grade": grade,
                "semester": semester,
                "courses": courses,
            }
        )

    # -----------------------------------------------------
    # 1. 질문에 학년 + 학기가 명확히 있는 경우
    # -----------------------------------------------------

    grade_match = re.search(
        r"([1-4])\s*학년",
        normalized_question,
    )

    semester_match = re.search(
        r"([1-2])\s*학기",
        normalized_question,
    )

    if grade_match and not semester_match:
        matches = [entry for entry in roadmap_entries if entry["grade"] == grade_match.group(1)]
        if not matches:
            return f"{grade_match.group(1)}학년 로드맵 과목은 현재 자료에서 확인되지 않습니다."
        return "\n".join(
            f"{entry['grade']}학년 {entry['semester']}학기: {', '.join(entry['courses'])}"
            for entry in matches
        )

    if (
        grade_match
        and semester_match
    ):
        target_grade = (
            grade_match.group(1)
        )

        target_semester = (
            semester_match.group(1)
        )

        for entry in roadmap_entries:
            if (
                entry["grade"]
                == target_grade
                and entry["semester"]
                == target_semester
            ):
                courses = ", ".join(
                    entry["courses"]
                )

                return (
                    f"{target_grade}학년 "
                    f"{target_semester}학기 과목은 "
                    f"{courses}입니다."
                )

    if grade_match and semester_match:
        return f"{grade_match.group(1)}학년 {semester_match.group(1)}학기 로드맵 과목은 현재 자료에서 확인되지 않습니다."

    # -----------------------------------------------------
    # 2. 질문 안에서 로드맵 과목명 직접 탐지
    # -----------------------------------------------------

    matched_courses = []

    for entry in roadmap_entries:
        for course in entry[
            "courses"
        ]:
            normalized_course = (
                normalize_text(
                    course
                )
            )

            compact_course = (
                normalized_course
                .replace(
                    " ",
                    "",
                )
            )

            if (
                compact_course
                and compact_course
                in compact_question
            ):
                matched_courses.append(
                    {
                        "course": course,
                        "grade": entry[
                            "grade"
                        ],
                        "semester": entry[
                            "semester"
                        ],
                    }
                )

    if matched_courses:
        matched_courses.sort(
            key=lambda item: len(
                item["course"]
            ),
            reverse=True,
        )

        matched = (
            matched_courses[0]
        )

        course = matched[
            "course"
        ]

        grade = matched[
            "grade"
        ]

        semester = matched[
            "semester"
        ]

        asks_track = any(
            keyword
            in normalized_question
            for keyword in [
                "무슨 트랙",
                "어느 트랙",
                "포함된 트랙",
                "어떤 트랙",
                "속해",
                "트랙 알려",
            ]
        )

        if asks_track:
            return (
                f"{course}은(는) "
                f"{track_name} 로드맵에 포함되어 있으며, "
                f"{grade}학년 {semester}학기에 "
                f"배치되어 있습니다."
            )

        return (
            f"{course}은(는) "
            f"{track_name} 로드맵에서 "
            f"{grade}학년 {semester}학기에 "
            f"나옵니다."
        )

    # -----------------------------------------------------
    # 3. 특정 과목을 물었지만 로드맵에 없는 경우
    # -----------------------------------------------------

    asks_specific_location = any(
        keyword
        in normalized_question
        for keyword in [
            "무슨 트랙",
            "어느 트랙",
            "어떤 트랙",
            "몇 학년",
            "몇학년",
            "몇 학기",
            "몇학기",
            "속해",
            "포함된 트랙",
            "로드맵에서",
        ]
    )

    if asks_specific_location:
        course_candidate = None

        candidate_patterns = [
            (
                r"^\s*(.+?)(?:은|는|이|가)?\s*"
                r"(?:무슨|어느|어떤)\s*트랙"
            ),
            (
                r"^\s*(.+?)(?:은|는|이|가)?\s*"
                r"로드맵에서"
            ),
            (
                r"^\s*(.+?)(?:은|는|이|가)?\s*"
                r"몇\s*학년"
            ),
            (
                r"^\s*(.+?)(?:은|는|이|가)?\s*"
                r"몇\s*학기"
            ),
        ]

        for pattern in candidate_patterns:
            candidate_match = re.search(
                pattern,
                question,
                flags=re.IGNORECASE,
            )

            if candidate_match:
                course_candidate = (
                    candidate_match
                    .group(1)
                    .strip()
                )

                course_candidate = re.sub(
                    r"(은|는|이|가)$",
                    "",
                    course_candidate,
                ).strip()

                break

        if course_candidate:
            return (
                f"{course_candidate}은(는) "
                f"현재 {track_name} 로드맵에서 "
                "확인되지 않습니다.\n"
                "공식 교과과정의 개설 과목과 "
                "로드맵의 추천·학습 경로는 "
                "서로 다를 수 있습니다."
            )

        return (
            "질문한 과목은 현재 "
            f"{track_name} 로드맵에서 "
            "확인되지 않습니다."
        )

    # -----------------------------------------------------
    # 4. 일반적인 전체 로드맵 질문
    # -----------------------------------------------------

    return None


# =========================================================
# 모델 / DB 로드
# =========================================================


def load_model() -> SentenceTransformer:
    """
    E5 임베딩 모델을 불러온다.
    """

    print(
        "임베딩 모델 불러오는 중..."
    )

    print(
        f"모델: {MODEL_NAME}"
    )

    # Fresh servers use the same verified public snapshot as the local app.
    model_options = {}
    if MODEL_NAME == "intfloat/multilingual-e5-small":
        model_options["revision"] = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    return SentenceTransformer(MODEL_NAME, **model_options)


def load_intent_classifier() -> Any:
    """
    step7에서 직접 학습한
    질문 의도 분류 모델을 불러온다.
    """

    if not INTENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            "질문 의도 분류 모델을 "
            "찾을 수 없습니다: "
            f"{INTENT_MODEL_PATH.resolve()}"
        )

    return joblib.load(
        INTENT_MODEL_PATH
    )


def load_collection():
    """
    기존 ChromaDB 컬렉션을 불러온다.
    """

    db_path = Path(
        CHROMA_DB_PATH
    )

    if not db_path.exists():
        raise FileNotFoundError(
            "ChromaDB 폴더를 찾을 수 없습니다: "
            f"{db_path.resolve()}"
        )

    client = (
        chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
        )
    )

    try:
        collection = (
            client.get_collection(
                name=COLLECTION_NAME,
            )
        )

    except Exception as error:
        raise RuntimeError(
            "컬렉션을 찾을 수 없습니다: "
            f"{COLLECTION_NAME}"
        ) from error

    if (
        collection.count()
        == 0
    ):
        raise ValueError(
            "저장된 문서 청크가 없습니다."
        )

    return collection


# =========================================================
# 교육과정 조건 추출
# =========================================================


def extract_curriculum_conditions(
    question: str,
) -> dict[str, str | int | None]:
    """
    질문에서 교육과정 검색 조건을 추출한다.
    """

    normalized_question = (
        normalize_text(
            question
        )
    )

    conditions: dict[
        str,
        str | int | None,
    ] = {
        "year": None,
        "grade": None,
        "semester": None,
        "completion_category": None,
        "required_type": None,
    }

    year_match = re.search(
        r"(20\d{2})\s*학년도",
        normalized_question,
    )

    if not year_match:
        year_match = re.search(
            r"(20\d{2})\s*년",
            normalized_question,
        )

    if year_match:
        conditions["year"] = (
            year_match.group(1)
        )

    grade_match = re.search(
        r"([1-4])\s*학년",
        normalized_question,
    )

    if grade_match:
        conditions["grade"] = int(
            grade_match.group(1)
        )

    semester_match = re.search(
        r"([1-2])\s*학기",
        normalized_question,
    )

    if semester_match:
        conditions["semester"] = int(
            semester_match.group(1)
        )

    if re.search(r"여름|하계", normalized_question):
        conditions["semester"] = 3
    elif re.search(r"겨울|동계", normalized_question):
        conditions["semester"] = 4
    elif re.search(r"첫(?:\s*번째)?\s*학기", normalized_question):
        conditions["semester"] = 1
    elif re.search(r"두\s*번째\s*학기", normalized_question):
        conditions["semester"] = 2

    if (
        "전공"
        in normalized_question
    ):
        conditions[
            "completion_category"
        ] = "전공"

    elif (
        "교양"
        in normalized_question
    ):
        conditions[
            "completion_category"
        ] = "교양"

    if (
        "필수"
        in normalized_question
    ):
        conditions[
            "required_type"
        ] = "필수"

    elif (
        "선택"
        in normalized_question
    ):
        conditions[
            "required_type"
        ] = "선택"

    return conditions


# =========================================================
# 기존 규칙 기반 질문 판별
# =========================================================


def is_explicit_roadmap_question(
    question: str,
) -> bool:
    """
    질문이 실제 교육과정이 아니라
    로드맵/트랙을 묻는 질문인지 판단한다.
    """

    normalized_question = (
        normalize_text(
            question
        )
    )

    roadmap_keywords = [
        "로드맵",
        "트랙",
        "ai·bigdata",
        "bigdata",
        "cloud",
        "web·app",
        "sw engineering",
    ]

    return any(
        keyword
        in normalized_question
        for keyword
        in roadmap_keywords
    )


def is_curriculum_question(
    question: str,
) -> bool:
    """
    기존 규칙 기반으로
    구조화 교육과정 질문인지 판단한다.
    """

    if (
        is_explicit_roadmap_question(
            question
        )
    ):
        return False

    normalized_question = (
        normalize_text(
            question
        )
    )

    if (
        "졸업"
        in normalized_question
    ):
        return False

    conditions = (
        extract_curriculum_conditions(
            question
        )
    )

    curriculum_keywords = [
        "과목",
        "교과목",
        "교육과정",
        "커리큘럼",
        "수업",
        "개설",
    ]

    has_curriculum_keyword = any(
        keyword
        in normalized_question
        for keyword
        in curriculum_keywords
    )

    has_grade_semester = (
        conditions["grade"]
        is not None
        or conditions["semester"]
        is not None
    )

    return (
        has_curriculum_keyword
        or has_grade_semester
    )


def create_unsupported_curriculum_message(
    question: str,
) -> str | None:
    """
    학교 공식 교과과정표에 없는
    4학년 교과과정 질문을 안내한다.

    명시적 로드맵 질문은 차단하지 않는다.
    """

    requested_intent = get_request_context(str(question))['request_intent']
    if requested_intent is not None and requested_intent != 'curriculum':
        return None
    if explicit_extension_intent(question):
        return None

    normalized_question = (
        normalize_text(
            question
        )
    )

    if (
        is_explicit_roadmap_question(
            question
        )
    ):
        return None

    grade_match = re.search(
        r"([1-4])\s*학년",
        normalized_question,
    )

    if not grade_match:
        return None

    grade = int(
        grade_match.group(1)
    )

    if grade != 4:
        return None

    curriculum_keywords = [
        "과목",
        "교과목",
        "교육과정",
        "커리큘럼",
        "수업",
        "개설",
        "전공",
        "교양",
        "필수",
        "선택",
    ]

    is_curriculum_request = any(
        keyword
        in normalized_question
        for keyword
        in curriculum_keywords
    )

    if not is_curriculum_request:
        return None

    return (
        "현재 공식 교과과정표에서 확인 가능한 "
        "AI소프트웨어학과 교과과정은 "
        "1~3학년까지입니다.\n"
        "4학년 교과과정 데이터는 "
        "학교 교과과정표에 제공되지 않습니다.\n"
        "단, 4학년 로드맵 정보는 "
        "별도로 검색할 수 있습니다."
    )


# =========================================================
# 교육과정 DB
# =========================================================


def load_all_curriculum_records(
    collection,
) -> list[dict]:
    """
    ChromaDB에서 교육과정 데이터를 모두 불러온다.
    """

    results = collection.get(
        where={
            "document_type": "교육과정"
        },
        include=[
            "documents",
            "metadatas",
        ],
    )

    ids = results.get(
        "ids",
        [],
    )

    documents = results.get(
        "documents",
        [],
    )

    metadatas = results.get(
        "metadatas",
        [],
    )

    records = []

    for (
        item_id,
        document,
        metadata,
    ) in zip(
        ids,
        documents,
        metadatas,
    ):
        record = {
            "id": item_id,
            "text": document,
            "source": metadata.get(
                "source",
                "알 수 없음",
            ),
            "page": metadata.get(
                "page",
                0,
            ),
            "chunk_id": metadata.get(
                "chunk_id",
                "알 수 없음",
            ),
            "document_type": (
                metadata.get(
                    "document_type",
                    "교육과정",
                )
            ),
            "year": metadata.get(
                "year",
                "unknown",
            ),
            "department": metadata.get(
                "department",
                "",
            ),
            "grade": metadata.get(
                "grade",
                0,
            ),
            "semester": metadata.get(
                "semester",
                0,
            ),
            "completion_type": (
                metadata.get(
                    "completion_type",
                    "",
                )
            ),
            "course_type": metadata.get(
                "course_type",
                "",
            ),
            "course_name": metadata.get(
                "course_name",
                "",
            ),
            "credit": metadata.get(
                "credit",
                0,
            ),
        }

        records.append(
            record
        )

    return records


def find_course_name_in_question(
    question: str,
    curriculum_records: list[dict],
) -> str | None:
    """
    질문 안에 실제 교과목명이 포함돼 있는지 확인한다.

    띄어쓰기 차이도 허용한다.
    """

    normalized_question = (
        normalize_text(
            question
        )
    )

    compact_question = (
        normalized_question
        .replace(
            " ",
            "",
        )
    )

    matches = []

    for record in (
        curriculum_records
    ):
        course_name = str(
            record.get(
                "course_name",
                "",
            )
        ).strip()

        if not course_name:
            continue

        normalized_course_name = (
            normalize_text(
                course_name
            )
        )

        compact_course_name = (
            normalized_course_name
            .replace(
                " ",
                "",
            )
        )

        exact_match = (
            normalized_course_name
            and normalized_course_name
            in normalized_question
        )

        compact_match = (
            compact_course_name
            and compact_course_name
            in compact_question
        )

        if (
            exact_match
            or compact_match
        ):
            matches.append(
                course_name
            )

    if not matches:
        return None

    matches.sort(
        key=len,
        reverse=True,
    )

    return matches[0]


# =========================================================
# AI 질문 의도 분류
# =========================================================


def predict_question_intent(
    intent_model: Any,
    question: str,
) -> dict[str, Any]:
    """
    직접 학습한 모델로 질문 의도를 예측한다.
    """

    predicted_intent = str(
        intent_model.predict(
            [question]
        )[0]
    )

    probabilities = (
        intent_model.predict_proba(
            [question]
        )[0]
    )

    classes = [
        str(class_name)
        for class_name
        in intent_model.classes_
    ]

    probability_map = {
        class_name: float(
            probability
        )
        for (
            class_name,
            probability,
        )
        in zip(
            classes,
            probabilities,
        )
    }

    confidence = max(
        probability_map.values()
    )

    return {
        "predicted_intent": (
            predicted_intent
        ),
        "confidence": (
            confidence
        ),
        "probabilities": (
            probability_map
        ),
    }


def is_known_unsupported_scope_question(
    question: str,
) -> bool:
    """
    현재 데이터가 없는
    명확한 미지원 학사 영역을 검사한다.
    """

    question = request_text(question)
    normalized_question = (
        normalize_text(
            question
        )
    )

    # Reject explicit contact/external-exam/message requests before entity overrides.
    # Application-form instructions (e.g. entering a contact number) stay eligible.
    compact_question = re.sub(r"\s+", "", str(question)).lower()
    requested = re.split(r"말고|아니고|아니라", compact_question)[-1]
    contact_request = re.search(r"(?:전화번호|연락처).*(?:알려|찾|문의|통화|어디|뭐)", requested)
    form_entry = ("신청서" in compact_question
                  and re.search(r"(?:전화번호|연락처).{0,10}(?:적|기입|작성)", compact_question)
                  and not re.search(r"담당|직원|조교|통화|문의", compact_question))
    if form_entry:
        contact_request = None
    exam_request = re.search(r"(?:시험|실기|필기).*(?:원서|접수사이트|홈페이지|사이트|링크)", requested)
    exam_admin_request = re.search(
        r"(?:응시료|시험장|합격발표).*(?:얼마|언제|어디|위치|날짜|일정|알려|찾|지원)|자격증발급(?:방법|절차|사이트)", requested)
    code_request = re.search(r"(?:과제|프로젝트|소스).*?코드.*?(?:작성|완성|고쳐|수정|디버깅|짜줘)", requested)
    message_request = re.search(r"(?:메시지|문장|메일|카카오톡).*(?:작성해|써줘|써주세요|써줄)", requested)
    if contact_request or exam_request or exam_admin_request or code_request or message_request:
        return True

    # These words occur in the new official eligibility rules. Other unsupported
    # domains (scholarships, tuition, etc.) still take precedence.
    allowed = {"수강신청", "휴학", "복학"} if explicit_extension_intent(question) else set()
    return any(
        keyword
        in normalized_question
        for keyword
        in UNSUPPORTED_SCOPE_KEYWORDS
        if keyword not in allowed
    )


def infer_expected_document_type(
    question: str,
) -> str | None:
    """
    기존 규칙 기반 문서 유형 추정.

    AI 모델의 신뢰도가 낮을 때
    fallback 용도로 사용한다.
    """

    normalized_question = (
        normalize_text(
            question
        )
    )

    if (
        is_explicit_roadmap_question(
            question
        )
    ):
        return "로드맵"

    graduation_keywords = [
        "졸업",
        "졸업학점",
        "졸업요건",
        "졸업기준",
        "최소취득학점",
    ]

    if any(
        keyword
        in normalized_question
        for keyword
        in graduation_keywords
    ):
        return "졸업요건"

    if (
        "학점"
        in normalized_question
        and (
            "전공"
            in normalized_question
            or "교양"
            in normalized_question
        )
        and "학년"
        not in normalized_question
        and "학기"
        not in normalized_question
        and "과목"
        not in normalized_question
    ):
        return "졸업요건"

    if (
        is_curriculum_question(
            question
        )
    ):
        return "교육과정"

    return None


def document_type_to_intent(
    document_type: str | None,
) -> str | None:
    """
    기존 문서 종류를
    의도 분류 라벨로 변환한다.
    """

    mapping = {
        "졸업요건": "graduation",
        "교육과정": "curriculum",
        "로드맵": "roadmap",
    }

    if document_type is None:
        return None

    return mapping.get(
        document_type
    )


def resolve_question_intent(
    question: str,
    collection,
    intent_model: Any,
) -> dict[str, Any]:
    """
    최종 질문 라우팅을 결정한다.

    우선순위:
    1. 명확한 미지원 영역
    2. 기존 입력 정책(되묻기/인사) 및 명확한 인정/대체 문맥
    3. 실제 교과목명 + 과목 속성 조회(또는 과목명 단독)
       단, 로드맵/트랙 문맥에서는 강제 curriculum 처리하지 않음
    4. 직접 학습한 AI 의도 분류 모델(confidence >= 0.50)
    5. 낮은 confidence에서는 기존 규칙 fallback, 불명확하면 out_of_scope
    """

    # -----------------------------------------------------
    # 1. 명확한 미지원 영역
    # -----------------------------------------------------

    if (
        is_known_unsupported_scope_question(
            question
        )
    ):
        return {
            "final_intent": (
                "out_of_scope"
            ),
            "routing_source": (
                "unsupported_scope_guard"
            ),
            "predicted_intent": None,
            "confidence": None,
            "detected_course_name": None,
        }

    # -----------------------------------------------------
    # 2. 신규 관계/자격증 문맥은 과목명 안전망보다 먼저 처리한다.
    # -----------------------------------------------------

    policy = input_policy(question)
    if policy:
        return {"predicted_intent": None, "confidence": None, "detected_course_name": None,
                "routing_source": "input_policy", **policy}

    context = get_request_context(str(question))
    if context['request_intent'] is not None:
        prediction = predict_question_intent(intent_model, question)
        return {**prediction, **context, 'final_intent': context['request_intent'],
                'routing_source': 'requested_object_guard', 'detected_course_name': None}

    extension_intent = explicit_extension_intent(question)
    if extension_intent:
        prediction = predict_question_intent(intent_model, question)
        return {
            **prediction, "final_intent": extension_intent,
            "routing_source": "extension_context_guard", "detected_course_name": None,
        }

    curriculum_records = (
        load_all_curriculum_records(
            collection
        )
    )

    detected_course_name = (
        find_course_name_in_question(
            question=question,
            curriculum_records=(
                curriculum_records
            ),
        )
    )

    # 중요:
    # 과목명이 있어도 질문이 명시적으로
    # 로드맵/트랙을 묻는 경우에는 curriculum으로 가로채지 않는다.
    compact_question = re.sub(r"\s+", "", str(question)).lower()
    course_attribute_request = re.search(
        r"학점|이수구분|전공[/·]교양|(?:어느|몇|언제).*?(?:학기|학년)|(?:학기|학년).*?(?:개설|편성|듣|배우)|(?:과목|수업)(?:정보|코드|자체)|개설|시수",
        compact_question,
    )
    course_name_only = (detected_course_name is not None and re.fullmatch(
        re.escape(re.sub(r"\s+", "", detected_course_name).lower())
        + r"(?:은|는|이|가|에대해|에대해서)?(?:뭐야|알려줘|알려주세요|설명해줘|설명해주세요)?[?!~.。！？]*",
        compact_question))
    if (
        detected_course_name is not None
        and (course_attribute_request or course_name_only)
        and not is_explicit_roadmap_question(
            question
        )
    ):
        return {
            "final_intent": (
                "curriculum"
            ),
            "routing_source": (
                "course_name_override"
            ),
            "predicted_intent": None,
            "confidence": None,
            "detected_course_name": (
                detected_course_name
            ),
        }

    # -----------------------------------------------------
    # 3. 직접 학습한 AI 모델
    # -----------------------------------------------------

    prediction = (
        predict_question_intent(
            intent_model=(
                intent_model
            ),
            question=question,
        )
    )

    predicted_intent = (
        prediction[
            "predicted_intent"
        ]
    )

    confidence = float(
        prediction[
            "confidence"
        ]
    )

    if (
        confidence
        >= INTENT_CONFIDENCE_THRESHOLD
    ):
        return {
            "final_intent": (
                predicted_intent
            ),
            "routing_source": (
                "intent_model"
            ),
            "predicted_intent": (
                predicted_intent
            ),
            "confidence": (
                confidence
            ),
            "probabilities": (
                prediction[
                    "probabilities"
                ]
            ),
            "detected_course_name": (
                detected_course_name
            ),
        }

    # -----------------------------------------------------
    # 4. 낮은 confidence → 규칙 기반 fallback
    # -----------------------------------------------------

    fallback_document_type = (
        infer_expected_document_type(
            question
        )
    )

    fallback_intent = (
        document_type_to_intent(
            fallback_document_type
        )
    )

    if (
        fallback_intent
        is not None
    ):
        return {
            "final_intent": (
                fallback_intent
            ),
            "routing_source": (
                "rule_fallback"
            ),
            "predicted_intent": (
                predicted_intent
            ),
            "confidence": (
                confidence
            ),
            "probabilities": (
                prediction[
                    "probabilities"
                ]
            ),
            "detected_course_name": (
                detected_course_name
            ),
        }

    # -----------------------------------------------------
    # 5. 최종 안전 처리
    # -----------------------------------------------------

    return {
        "final_intent": (
            "out_of_scope"
        ),
        "routing_source": (
            "safe_fallback"
        ),
        "predicted_intent": (
            predicted_intent
        ),
        "confidence": (
            confidence
        ),
        "probabilities": (
            prediction[
                "probabilities"
            ]
        ),
        "detected_course_name": (
            detected_course_name
        ),
    }


def retrieval_question_for_decision(question, decision, collection):
    """Use an original request span; retain a uniquely referenced course name.

    Routing still predicts from the original input. Context selection prevents
    negated/background entities from becoming structured lookup conditions.
    """
    if decision.get('routing_source') != 'requested_object_guard':
        return question
    focused = decision.get('request_text', question)
    if focused == question:
        return question
    intent = decision['final_intent']
    if intent in {'curriculum', 'roadmap'}:
        records = load_all_curriculum_records(collection)
        focused_name = find_course_name_in_question(focused, records)
        reference = re.search(r'이\s*(?:과목|수업|공부)|그\s*(?:과목|수업|공부|걸)|정규\s*수업|강의\s*자체', focused)
        if not focused_name and reference:
            names = {r['course_name'] for r in records
                     if find_course_name_in_question(question, [r])}
            # Prefer a full course name over its substring (C vs C/C++).
            names = {n for n in names if not any(n != m and normalize_text(n).replace(' ', '')
                     in normalize_text(m).replace(' ', '') for m in names)}
            if len(names) == 1:
                focused = next(iter(names)) + ' ' + focused
    elif intent == 'graduation':
        # Admission/program constraints remain applicable across a background clause.
        for pattern in [r'[23]\s*년제', r'편입학?|신입학?', r'20\d{2}\s*학번', r'전공심화']:
            qualifiers = set(re.findall(pattern, question))
            if len(qualifiers) == 1 and not re.search(pattern, focused):
                focused = next(iter(qualifiers)) + ' ' + focused
    return focused


# =========================================================
# 교육과정 구조화 검색
# =========================================================


def search_curriculum_structured(
    collection,
    question: str,
    force: bool = False,
) -> list[dict]:
    """
    교육과정 메타데이터를 사용해 정확한 과목을 찾는다.

    force=True이면 AI 의도 분류 결과가 curriculum인 경우
    기존 규칙 판별을 거치지 않고 구조화 검색을 수행한다.
    """

    curriculum_records = (
        load_all_curriculum_records(
            collection
        )
    )

    if not curriculum_records:
        return []

    conditions = (
        extract_curriculum_conditions(
            question
        )
    )

    detected_course_name = (
        find_course_name_in_question(
            question=question,
            curriculum_records=(
                curriculum_records
            ),
        )
    )

    if (
        not force
        and not is_curriculum_question(
            question
        )
        and detected_course_name
        is None
    ):
        return []

    if detected_course_name is None:
        named = re.search(r"(\S+)\s+과목(?:은|는|의|이|가|을)", question)
        generic = {"전공", "교양", "필수", "선택", "개설", "전체", "모든", "어떤", "무슨", "교과과정", "교육과정", "개설된", "있는", "배우는", "듣는", "이수하는", "수강하는"}
        # Completion categories are generic list filters, not missing subject names.
        generic.add("포함된")
        generic.update(str(record.get("completion_type", "")).split("(")[0] for record in curriculum_records)
        if named and named.group(1) not in generic and not re.search(r"학년|학기", named.group(1)):
            return []
        if not any(conditions.values()) and re.search(r"몇\s*학점|개설\s*학기", question):
            return []

    filtered_records = []

    for record in (
        curriculum_records
    ):
        if (
            conditions["year"]
            is not None
            and str(
                record["year"]
            )
            != str(
                conditions["year"]
            )
        ):
            continue

        if (
            conditions["grade"]
            is not None
            and int(
                record["grade"]
            )
            != conditions["grade"]
        ):
            continue

        if (
            conditions["semester"]
            is not None
            and int(
                record["semester"]
            )
            != conditions["semester"]
        ):
            continue

        completion_type = str(
            record[
                "completion_type"
            ]
        )

        completion_category = (
            conditions[
                "completion_category"
            ]
        )

        if (
            completion_category
            is not None
            and str(
                completion_category
            )
            not in completion_type
        ):
            continue

        required_type = (
            conditions[
                "required_type"
            ]
        )

        if (
            required_type
            is not None
            and str(
                required_type
            )
            not in completion_type
        ):
            continue

        if (
            detected_course_name
            is not None
            and record[
                "course_name"
            ]
            != detected_course_name
        ):
            continue

        filtered_records.append(
            record
        )

    filtered_records.sort(
        key=lambda item: (
            int(
                item.get(
                    "grade",
                    0,
                )
            ),
            int(
                item.get(
                    "semester",
                    0,
                )
            ),
            int(
                item.get(
                    "chunk_id",
                    0,
                )
            )
            if str(
                item.get(
                    "chunk_id",
                    "",
                )
            ).isdigit()
            else 999999,
        )
    )

    return filtered_records


def format_semester(semester: int | str) -> str:
    """Translate the source metadata's seasonal codes into student-facing labels."""
    return {"1": "1학기", "2": "2학기", "3": "여름계절학기", "4": "겨울계절학기"}.get(str(semester), "학기 미확인")


def create_curriculum_answer(
    question: str,
    results: list[dict],
) -> str:
    """
    구조화 교육과정 검색 결과를
    자연스러운 목록으로 변환한다.
    """

    if not results:
        return (
            "조건에 맞는 교육과정 "
            "과목을 찾지 못했습니다."
        )

    conditions = (
        extract_curriculum_conditions(
            question
        )
    )

    year = (
        conditions["year"]
    )

    grade = (
        conditions["grade"]
    )

    semester = (
        conditions["semester"]
    )

    unique_course_names = {
        result["course_name"]
        for result
        in results
    }

    asks_credit = (
        "학점"
        in normalize_text(
            question
        )
    )

    if (
        len(results) == 1
        and (
            asks_credit
            or len(
                unique_course_names
            ) == 1
        )
    ):
        result = results[0]

        answer_lines = [
            (
                f"{result['course_name']}은(는) "
                f"{result['grade']}학년 "
                f"{format_semester(result['semester'])}에 "
                f"{result['credit']}학점으로 "
                "개설되어 있습니다."
            ),
            (
                "이수구분은 "
                f"{result['completion_type']}입니다."
            ),
            "",
            "[근거]",
            (
                "- 학과: "
                f"{result['department']}"
            ),
            (
                "- 학년도: "
                f"{result['year']}"
            ),
            (
                "- 문서: "
                f"{result['source']}"
            ),
        ]

        if asks_credit:
            answer_lines[:2] = [f"{result['course_name']}은(는) {result['credit']}학점입니다."]
        return "\n".join(
            answer_lines
        )

    heading_parts = []

    if year:
        heading_parts.append(
            f"{year}학년도"
        )

    if grade:
        heading_parts.append(
            f"{grade}학년"
        )

    if semester:
        heading_parts.append(
            format_semester(semester)
        )

    if (
        conditions[
            "completion_category"
        ]
    ):
        heading_parts.append(
            str(
                conditions[
                    "completion_category"
                ]
            )
        )

    if (
        conditions[
            "required_type"
        ]
    ):
        heading_parts.append(
            str(
                conditions[
                    "required_type"
                ]
            )
        )

    heading = " ".join(
        heading_parts
    )

    if not heading:
        heading = "해당"

    answer_lines = [
        (
            f"{heading} 과목은 "
            f"총 {len(results)}개입니다."
        )
    ]

    total_credits = 0

    for result in results:
        credit = int(
            result.get(
                "credit",
                0,
            )
        )

        total_credits += credit

        answer_lines.append(
            f"- {result['course_name']} "
            f"({credit}학점, "
            f"{result['completion_type']})"
        )

    answer_lines.extend(
        [
            "",
            (
                "개설 과목 학점 단순 합계: "
                f"{total_credits}학점"
            ),
            "",
            "[근거]",
            (
                "- 문서: "
                f"{results[0]['source']}"
            ),
            (
                "- 문서 종류: "
                f"{results[0]['document_type']}"
            ),
            (
                "- 학년도: "
                f"{results[0]['year']}"
            ),
        ]
    )

    return "\n".join(
        answer_lines
    )


# =========================================================
# 질문 키워드
# =========================================================


def extract_question_keywords(
    question: str,
) -> list[str]:
    """
    검색에 유용한 질문 핵심 단어를 추출한다.
    """

    normalized_question = (
        normalize_text(
            question
        )
    )

    stopwords = {
        "내용",
        "알려줘",
        "알려주세요",
        "무엇인가요",
        "뭐가",
        "뭐야",
        "대해",
        "대한",
        "입니다",
        "있나요",
        "설명해줘",
        "설명해주세요",
        "관련",
        "과목은",
        "학점은",
        "몇",
    }

    words = (
        normalized_question
        .split()
    )

    keywords = []

    for word in words:
        if len(word) < 2:
            continue

        if word in stopwords:
            continue

        if word not in keywords:
            keywords.append(
                word
            )

    return keywords


# =========================================================
# E5 벡터 검색
# =========================================================


@lru_cache(maxsize=128)
def _cached_query_embedding(model: SentenceTransformer, question: str) -> tuple[float, ...]:
    # Cache is process-local and keyed by the loaded model object and exact trimmed text.
    # A replaced model gets a distinct key; clear the cache if mutating weights in place.
    embedding = model.encode(
        f"query: {question}", convert_to_numpy=True, normalize_embeddings=True,
    )
    return tuple(embedding.tolist())


def create_query_embedding(
    model: SentenceTransformer,
    question: str,
) -> list[float]:
    """Return an independent list so callers cannot mutate the cached embedding."""
    return list(_cached_query_embedding(model, question.strip()))


def search_documents(
    collection,
    query_embedding: list[float],
) -> list[dict]:
    """
    의미적으로 유사한 문서 후보를 검색한다.
    """

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        where={"domain": "legacy"},
        n_results=(
            SEARCH_RESULT_COUNT
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        results.get(
            "documents",
            [[]],
        )[0]
    )

    metadatas = (
        results.get(
            "metadatas",
            [[]],
        )[0]
    )

    distances = (
        results.get(
            "distances",
            [[]],
        )[0]
    )

    search_results = []

    for (
        document,
        metadata,
        distance,
    ) in zip(
        documents,
        metadatas,
        distances,
    ):
        semantic_similarity = max(
            0.0,
            1.0
            - float(
                distance
            ),
        )

        search_results.append(
            {
                "text": document,
                "source": (
                    metadata.get(
                        "source",
                        "알 수 없음",
                    )
                ),
                "page": (
                    metadata.get(
                        "page",
                        "알 수 없음",
                    )
                ),
                "chunk_id": (
                    metadata.get(
                        "chunk_id",
                        "알 수 없음",
                    )
                ),
                "document_type": (
                    metadata.get(
                        "document_type",
                        "기타",
                    )
                ),
                "year": (
                    metadata.get(
                        "year",
                        "unknown",
                    )
                ),
                "department": (
                    metadata.get(
                        "department",
                        "",
                    )
                ),
                "grade": (
                    metadata.get(
                        "grade",
                        0,
                    )
                ),
                "semester": (
                    metadata.get(
                        "semester",
                        0,
                    )
                ),
                "completion_type": (
                    metadata.get(
                        "completion_type",
                        "",
                    )
                ),
                "course_type": (
                    metadata.get(
                        "course_type",
                        "",
                    )
                ),
                "course_name": (
                    metadata.get(
                        "course_name",
                        "",
                    )
                ),
                "credit": (
                    metadata.get(
                        "credit",
                        0,
                    )
                ),
                "semantic_similarity": (
                    semantic_similarity
                ),
            }
        )

    return search_results


def calculate_keyword_score(
    question: str,
    document_text: str,
) -> tuple[float, list[str]]:
    """
    질문 키워드가 문서에 얼마나
    포함되어 있는지 계산한다.
    """

    keywords = (
        extract_question_keywords(
            question
        )
    )

    if not keywords:
        return 0.0, []

    normalized_document = (
        normalize_text(
            document_text
        )
    )

    matched_keywords = [
        keyword
        for keyword
        in keywords
        if keyword
        in normalized_document
    ]

    keyword_score = (
        len(
            matched_keywords
        )
        / len(
            keywords
        )
    )

    return (
        keyword_score,
        matched_keywords,
    )


def calculate_document_type_bonus(
    question: str,
    document_type: str,
    expected_document_type_override: str | None = None,
) -> tuple[float, str | None]:
    """
    질문 의도와 문서 종류가 일치하면
    검색 점수 보너스를 부여한다.

    AI 모델이 의도를 결정한 경우에는
    expected_document_type_override를 사용한다.
    """

    if (
        expected_document_type_override
        is not None
    ):
        expected_type = (
            expected_document_type_override
        )

    else:
        expected_type = (
            infer_expected_document_type(
                question
            )
        )

    if expected_type is None:
        return 0.0, None

    if (
        document_type
        == expected_type
    ):
        return (
            0.20,
            expected_type,
        )

    return (
        0.0,
        expected_type,
    )


def rerank_results(
    question: str,
    results: list[dict],
    expected_document_type_override: str | None = None,
) -> list[dict]:
    """
    의미 유사도 + 키워드 + 문서 종류를
    이용해 검색 결과를 재정렬한다.
    """

    reranked_results = []

    for result in results:
        (
            keyword_score,
            matched_keywords,
        ) = calculate_keyword_score(
            question=question,
            document_text=(
                result["text"]
            ),
        )

        (
            document_type_bonus,
            expected_document_type,
        ) = (
            calculate_document_type_bonus(
                question=question,
                document_type=(
                    result[
                        "document_type"
                    ]
                ),
                expected_document_type_override=(
                    expected_document_type_override
                ),
            )
        )

        final_score = (
            result[
                "semantic_similarity"
            ]
            * 0.75
            + keyword_score
            * 0.15
            + document_type_bonus
        )

        reranked_results.append(
            {
                **result,
                "keyword_score": (
                    keyword_score
                ),
                "matched_keywords": (
                    matched_keywords
                ),
                "document_type_bonus": (
                    document_type_bonus
                ),
                "expected_document_type": (
                    expected_document_type
                ),
                "final_score": (
                    final_score
                ),
            }
        )

    expected = expected_document_type_override or infer_expected_document_type(question)
    for result in reranked_results:
        text = result["text"]
        priority = False
        if expected == "졸업요건" and result["document_type"] == "졸업요건":
            if "전공심화" in question:
                priority = "전공심화" in text
            elif "편입" in question:
                priority = text.lstrip().startswith("편입학")
            else:
                priority = "신입학" in text
        result["academic_context_priority"] = priority
    reranked_results.sort(
        key=lambda result: (result["academic_context_priority"], result["final_score"]),
        reverse=True,
    )

    return reranked_results


def remove_duplicate_results(
    results: list[dict],
) -> list[dict]:
    """
    같은 내용의 검색 결과 중복을 제거한다.
    """

    unique_results = []
    seen_texts = set()

    for result in results:
        normalized = (
            normalize_text(
                result["text"]
            )
        )

        if normalized in seen_texts:
            continue

        seen_texts.add(
            normalized
        )

        unique_results.append(
            result
        )

    return unique_results


# =========================================================
# 최종 답변 생성
# =========================================================


def create_answer(
    question: str,
    results: list[dict],
) -> str:
    """
    벡터 검색 결과를 기반으로
    최종 답변을 생성한다.
    """

    if not results:
        return (
            "관련 문서를 찾지 못했습니다."
        )

    best_result = (
        results[0]
    )

    if (
        best_result[
            "final_score"
        ]
        < MIN_FINAL_SCORE
    ):
        return (
            "질문과 충분히 관련된 "
            "문서를 찾지 못했습니다.\n"
            "질문을 더 구체적으로 "
            "입력해 주세요."
        )

    focused_answer = None

    if (
        best_result[
            "document_type"
        ]
        == "졸업요건"
    ):
        focused_answer = (
            extract_academic_credit_answer(
                question=question,
                document_text=(
                    best_result[
                        "text"
                    ]
                ),
            )
        )

    elif (
        best_result[
            "document_type"
        ]
        == "로드맵"
    ):
        focused_answer = (
            extract_roadmap_answer(
                question=question,
                document_text=(
                    best_result[
                        "text"
                    ]
                ),
            )
        )

    answer_text = (
        focused_answer
        or best_result[
            "text"
        ]
    )

    return "\n".join([
        answer_text, "", "[근거]", f"- 문서: {best_result['source']}",
        f"- 페이지: {best_result['page']}",
    ])


def display_additional_sources(
    results: list[dict],
) -> None:
    """
    두 번째 이후 검색 결과를 표시한다.
    """

    if len(results) <= 1:
        return

    print()

    print(
        "=== 추가 참고 문서 ==="
    )

    for (
        index,
        result,
    ) in enumerate(
        results[1:4],
        start=1,
    ):
        print(
            f"{index}. "
            f"{result['source']} / "
            f"{result['document_type']} / "
            f"{result['page']}페이지 / "
            f"청크 {result['chunk_id']} / "
            f"최종 점수 "
            f"{result['final_score']:.4f}"
        )


def display_intent_decision(
    decision: dict[str, Any],
) -> None:
    """
    CLI에서 최종 질문 의도 판단 과정을 표시한다.
    """

    print()

    print(
        "=== 질문 의도 판단 ==="
    )

    # Keep the routing key unchanged; describe the existing response state separately.
    if decision.get("response_kind") == "clarification":
        print("응답 상태: 미확정/추가 질문 필요")
    else:
        print(
            "최종 의도: "
            f"{decision['final_intent']}"
        )

    routing_source = (
        decision[
            "routing_source"
        ]
    )

    source_names = {
        "intent_model": (
            "직접 학습 모델"
        ),
        "rule_fallback": (
            "규칙 기반 fallback"
        ),
        "course_name_override": (
            "실제 교과목명 안전망"
        ),
        "unsupported_scope_guard": (
            "미지원 영역 안전망"
        ),
        "safe_fallback": (
            "최종 안전 fallback"
        ),
    }

    print(
        "판단 방식: "
        f"{source_names.get(
            routing_source,
            routing_source,
        )}"
    )

    predicted_intent = (
        decision.get(
            "predicted_intent"
        )
    )

    confidence = (
        decision.get(
            "confidence"
        )
    )

    if (
        predicted_intent
        is not None
    ):
        print(
            "모델 예측: "
            f"{predicted_intent}"
        )

    if (
        confidence
        is not None
    ):
        print(
            "모델 신뢰도: "
            f"{float(confidence):.4f}"
        )

    detected_course_name = (
        decision.get(
            "detected_course_name"
        )
    )

    if (
        detected_course_name
        is not None
    ):
        print(
            "감지 교과목: "
            f"{detected_course_name}"
        )


# =========================================================
# Main
# =========================================================


def main() -> None:
    """
    프로그램 전체 실행 순서.
    """

    try:
        model = load_model()

        collection = (
            load_collection()
        )

        intent_model = (
            load_intent_classifier()
        )

        print(
            f"저장된 청크 수: "
            f"{collection.count()}"
        )

        print(
            "질문 의도 모델: "
            f"{INTENT_MODEL_PATH}"
        )

        print(
            "의도 모델 confidence threshold: "
            f"{INTENT_CONFIDENCE_THRESHOLD:.2f}"
        )

        print()

        print(
            "=== AI 학사 문서 답변 테스트 ==="
        )

        print(
            "종료하려면 exit 입력"
        )

        while True:
            print()

            question = input(
                "질문: "
            ).strip()

            if (
                question.lower()
                == "exit"
            ):
                print(
                    "프로그램을 종료합니다."
                )

                break

            if not question:
                print(
                    "질문을 입력해 주세요."
                )

                continue

            # -------------------------------------------------
            # 0. 공식 교과과정 미지원 범위
            # -------------------------------------------------

            unsupported_message = (
                create_unsupported_curriculum_message(
                    question
                )
            )

            if (
                unsupported_message
                is not None
            ):
                print()

                print(
                    "=== 답변 ==="
                )

                print(
                    unsupported_message
                )

                continue

            # -------------------------------------------------
            # 1. AI + fallback 기반 질문 의도 판단
            # -------------------------------------------------

            decision = (
                resolve_question_intent(
                    question=question,
                    collection=collection,
                    intent_model=(
                        intent_model
                    ),
                )
            )

            display_intent_decision(
                decision
            )

            final_intent = (
                decision[
                    "final_intent"
                ]
            )

            if decision.get("response_message"):
                print("\n=== 답변 ===\n" + decision["response_message"])
                continue

            # -------------------------------------------------
            # 2. 범위 밖 질문
            # -------------------------------------------------

            if (
                final_intent
                == "out_of_scope"
            ):
                print()

                print(
                    "=== 답변 ==="
                )

                print(
                    out_of_scope_answer(question)
                )

                continue

            # -------------------------------------------------
            # 3. 교육과정 구조화 검색
            # -------------------------------------------------

            if final_intent in {"substitute_course", "certificate_credit"}:
                extended = answer_extension(question, final_intent, collection, model)
                print("\n=== 답변 ===\n" + extended.get("display_answer", extended["answer"]))
                for evidence in extended["evidence"]:
                    print(f"근거: {evidence['source']} · {evidence['page']}쪽 · 2026학년도")
                continue

            if (
                final_intent
                == "curriculum"
            ):
                curriculum_results = (
                    search_curriculum_structured(
                        collection=collection,
                        question=question,
                        force=True,
                    )
                )

                print()

                print(
                    "=== 답변 ==="
                )

                print(
                    create_curriculum_answer(
                        question=question,
                        results=(
                            curriculum_results
                        ),
                    )
                )

                continue

            # -------------------------------------------------
            # 4. 졸업요건 / 로드맵 벡터 검색
            # -------------------------------------------------

            query_embedding = (
                create_query_embedding(
                    model=model,
                    question=question,
                )
            )

            results = (
                search_documents(
                    collection=collection,
                    query_embedding=(
                        query_embedding
                    ),
                )
            )

            results = (
                remove_duplicate_results(
                    results
                )
            )

            expected_document_type = (
                INTENT_TO_DOCUMENT_TYPE.get(
                    final_intent
                )
            )

            results = (
                rerank_results(
                    question=question,
                    results=results,
                    expected_document_type_override=(
                        expected_document_type
                    ),
                )
            )

            answer = (
                create_answer(
                    question=question,
                    results=results,
                )
            )

            print()

            print(
                "=== 답변 ==="
            )

            print(
                answer
            )

            display_additional_sources(
                results
            )

    except KeyboardInterrupt:
        print()

        print(
            "사용자가 프로그램을 종료했습니다."
        )

    except Exception as error:
        print()

        print(
            "답변 처리 중 오류가 발생했습니다."
        )

        print(
            f"오류 종류: "
            f"{type(error).__name__}"
        )

        print(
            f"오류 내용: "
            f"{error}"
        )


if __name__ == "__main__":
    main()
