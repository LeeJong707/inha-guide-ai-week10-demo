"""
7단계: Streamlit 웹 UI

기능:
- 웹 화면에서 질문 입력
- 직접 학습한 질문 의도 분류 모델 사용
- 모델 신뢰도가 낮으면 기존 규칙 기반 fallback 사용
- 실제 교과목명 안전망 적용
- 미지원 영역 안전망 적용
- 교육과정 질문은 구조화 메타데이터 검색
- 졸업요건 / 로드맵 질문은 로컬 임베딩 기반 검색
- 근거 문서와 검색 정보를 표시
- 외부 생성형 AI API 사용 안 함

실행:
python -m streamlit run app.py
"""

import logging
from pathlib import Path
import streamlit as st
from expansion_search import NEW_INTENTS, answer_extension, load_data

from step6_retrieval_answer import (
    INTENT_TO_DOCUMENT_TYPE,
    MIN_FINAL_SCORE,
    create_answer,
    format_semester,
    out_of_scope_answer,
    create_curriculum_answer,
    create_query_embedding,
    create_unsupported_curriculum_message,
    extract_academic_credit_answer,
    extract_roadmap_answer,
    load_collection,
    load_intent_classifier,
    load_model,
    remove_duplicate_results,
    rerank_results,
    resolve_question_intent,
    retrieval_question_for_decision,
    search_curriculum_structured,
    search_documents,
)


# =========================================================
# Streamlit 기본 설정
# =========================================================

st.set_page_config(
    page_title="AI 학사 문서 도우미",
    page_icon="🎓",
    layout="centered",
)


# =========================================================
# 화면 스타일
# =========================================================


def apply_custom_style() -> None:
    """
    기본 Streamlit 느낌을 조금 줄이고
    학사정보 검색 서비스에 맞는 단정한 UI를 적용한다.

    검색 / 모델 / DB 로직에는 영향을 주지 않는다.
    """

    st.markdown(
        """
        <style>

        /* -------------------------------------------------
           전체 화면
        ------------------------------------------------- */

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        /* -------------------------------------------------
           제목
        ------------------------------------------------- */

        .main-title {
            font-size: 2.6rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            color: #24272d;
            margin-bottom: 0.25rem;
        }

        .main-description {
            font-size: 1rem;
            color: #73777f;
            margin-bottom: 1.4rem;
            line-height: 1.6;
        }

        /* -------------------------------------------------
           시스템 설명
        ------------------------------------------------- */

        .system-description {
            font-size: 0.92rem;
            color: #5f666c;
            line-height: 1.65;
            padding: 0.8rem 0;
            margin-bottom: 0.7rem;
            border-top: 1px solid #e6e9eb;
            border-bottom: 1px solid #e6e9eb;
        }

        /* -------------------------------------------------
           DB 상태
        ------------------------------------------------- */

        .database-status {
            display: inline-block;
            padding: 0.45rem 0.75rem;
            margin-top: 0.2rem;
            margin-bottom: 1.3rem;

            border-radius: 8px;
            border: 1px solid #d8e7e7;

            background-color: #f5fafa;
            color: #446464;

            font-size: 0.9rem;
            font-weight: 500;
        }

        .database-status-dot {
            color: #69a8a8;
            margin-right: 0.25rem;
        }

        /* -------------------------------------------------
           입력창
        ------------------------------------------------- */

        div[data-testid="stTextInput"] input {
            border-radius: 8px;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #91c8ca;
            box-shadow: 0 0 0 1px #91c8ca;
        }

        /* -------------------------------------------------
           검색 버튼
        ------------------------------------------------- */

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] button {
            width: 100%;

            background-color: #9fd7d9 !important;
            color: #203638 !important;

            border: 1px solid #9fd7d9 !important;
            border-radius: 8px !important;

            font-weight: 600 !important;

            height: 2.8rem;

            transition: 0.15s ease;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #8bcacc !important;
            border-color: #8bcacc !important;
            color: #203638 !important;
        }

        div.stButton > button:active,
        div[data-testid="stFormSubmitButton"] button:active {
            background-color: #7fc0c2 !important;
            border-color: #7fc0c2 !important;
        }

        div.stButton > button:focus,
        div[data-testid="stFormSubmitButton"] button:focus {
            box-shadow: none !important;
        }

        /* -------------------------------------------------
           구분선
        ------------------------------------------------- */

        hr {
            border-color: #eceeef;
        }

        /* -------------------------------------------------
           Expander
        ------------------------------------------------- */

        div[data-testid="stExpander"] {
            border-radius: 8px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 모델 / DB 초기화
# =========================================================


@st.cache_resource
def initialize_search_system():
    """
    임베딩 모델, ChromaDB,
    질문 의도 분류 모델을 한 번만 불러온다.

    Streamlit은 사용자 입력 때마다 코드를 다시 실행하므로
    cache_resource를 사용해 모델과 DB를 재사용한다.
    """

    model = load_model()

    collection = load_collection()

    intent_model = (
        load_intent_classifier()
    )

    return (
        model,
        collection,
        intent_model,
    )


# =========================================================
# 질문 의도 판단 결과 표시
# =========================================================


def display_intent_decision(
    decision: dict,
) -> None:
    """
    질문 의도 분류 과정과
    최종 라우팅 결과를 접을 수 있는 영역에 표시한다.
    """

    source_names = {
        "requested_object_guard": "실제 요청 대상과 배경 문맥 구분",
        "extension_context_guard": "대체과목·자격증 문맥 확인",
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

    with st.expander(
        "개발 정보 보기"
    ):
        final_intent = decision.get(
            "final_intent",
            "알 수 없음",
        )

        # Clarification is unresolved even when the compatibility route is out_of_scope.
        if decision.get("response_kind") == "clarification":
            st.write("**응답 상태:** 미확정/추가 질문 필요")
        else:
            st.write(
                f"**최종 의도:** "
                f"{final_intent}"
            )

        routing_source = (
            decision.get(
                "routing_source",
                "알 수 없음",
            )
        )

        st.write(
            f"**판단 방식:** "
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

        detected_course_name = (
            decision.get(
                "detected_course_name"
            )
        )

        if (
            predicted_intent
            is not None
        ):
            st.write(
                f"**모델 예측:** "
                f"{predicted_intent}"
            )

        if (
            confidence
            is not None
        ):
            st.write(
                f"**모델 신뢰도:** "
                f"{float(confidence):.4f}"
            )

        if (
            detected_course_name
            is not None
        ):
            st.write(
                f"**감지 교과목:** "
                f"{detected_course_name}"
            )


# =========================================================
# 교육과정 결과 표시
# =========================================================


def display_curriculum_results(
    question: str,
    results: list[dict],
) -> None:
    """
    구조화 교육과정 검색 결과를 표시한다.
    """

    st.subheader(
        "답변"
    )

    answer = (
        create_curriculum_answer(
            question=question,
            results=results,
        )
    )

    st.markdown(answer)

    st.divider()

    st.subheader(
        "검색 조건"
    )

    first_result = (
        results[0]
    )

    st.write(
        f"**학과:** "
        f"{first_result.get(
            'department',
            '알 수 없음',
        )}"
    )

    st.write(
        f"**학년도:** "
        f"{first_result.get(
            'year',
            '알 수 없음',
        )}"
    )

    st.write(
        f"**문서 종류:** "
        f"{first_result.get(
            'document_type',
            '교육과정',
        )}"
    )

    st.write(
        f"**검색된 과목 수:** "
        f"{len(results)}개"
    )

    st.markdown("[교과과정 원본 페이지](https://cs.inhatc.ac.kr/cs/1768/subview.do)")
    st.caption("현재 답변은 저장된 2026학년도 자료 기준입니다. 원본 페이지의 이후 변경은 자동 반영되지 않습니다.")

    with st.expander(
        "검색된 교육과정 데이터 보기"
    ):
        for (
            index,
            result,
        ) in enumerate(
            results,
            start=1,
        ):
            st.markdown(
                f"### {index}. "
                f"{result.get(
                    'course_name',
                    '과목명 없음',
                )}"
            )

            st.write(
                f"**학년:** "
                f"{result.get(
                    'grade',
                    '?',
                )}학년"
            )

            st.write(f"**학기:** {format_semester(result.get('semester', '?'))}")

            st.write(
                f"**학점:** "
                f"{result.get(
                    'credit',
                    '?',
                )}학점"
            )

            st.write(
                f"**이수구분:** "
                f"{result.get(
                    'completion_type',
                    '알 수 없음',
                )}"
            )

            st.write(
                f"**교과구분:** "
                f"{result.get(
                    'course_type',
                    '알 수 없음',
                )}"
            )

            if (
                index
                < len(results)
            ):
                st.divider()


# =========================================================
# 벡터 검색 최상위 결과 표시
# =========================================================


def display_vector_best_result(
    question: str,
    result: dict,
) -> None:
    """
    일반 벡터 검색의 최상위 결과를 표시한다.
    """

    st.subheader(
        "답변"
    )

    focused_answer = None

    if (
        result.get(
            "document_type"
        )
        == "졸업요건"
    ):
        focused_answer = (
            extract_academic_credit_answer(
                question=question,
                document_text=(
                    result["text"]
                ),
            )
        )

    elif (
        result.get(
            "document_type"
        )
        == "로드맵"
    ):
        focused_answer = (
            extract_roadmap_answer(
                question=question,
                document_text=(
                    result["text"]
                ),
            )
        )

    if focused_answer:
        st.write(
            focused_answer
        )

        with st.expander(
            "검색된 원문 보기"
        ):
            st.write(
                result["text"]
            )

    else:
        st.write(
            result["text"]
        )

    st.divider()

    st.subheader(
        "근거 문서"
    )

    st.write(
        f"**문서:** "
        f"{result.get(
            'source',
            '알 수 없음',
        )}"
    )

    st.write(
        f"**문서 종류:** "
        f"{result.get(
            'document_type',
            '알 수 없음',
        )}"
    )

    source_year = result.get("year")
    st.write(f"**자료 연도:** {source_year if source_year not in (None, '', 'unknown') else '미표기 (원문 확인)'}")

    st.write(
        f"**페이지:** "
        f"{result.get(
            'page',
            '알 수 없음',
        )}"
    )

    with st.expander("개발 정보 · 검색 상세"):
        st.write(
            f"**청크 ID:** "
            f"{result.get(
                'chunk_id',
                '알 수 없음',
            )}"
        )

        st.write(
            f"**의미 유사도:** "
            f"{result[
                'semantic_similarity'
            ]:.4f}"
        )

        # 확률이 아니라 검색 재정렬용 점수이므로
        # UI에서는 "재정렬 점수"라고 표시한다.
        st.write(
            f"**재정렬 점수:** "
            f"{result[
                'final_score'
            ]:.4f}"
        )

        if (
            result.get(
                "document_type_bonus",
                0,
            )
            > 0
        ):
            st.write(
                f"**문서 종류 일치:** "
                f"{result.get(
                    'expected_document_type'
                )}"
            )

        matched_keywords = (
            result.get(
                "matched_keywords",
                [],
            )
        )

        if matched_keywords:
            st.write(
                "**일치 키워드:** "
                + ", ".join(
                    matched_keywords
                )
            )


# =========================================================
# 추가 참고 문서 표시
# =========================================================


def display_additional_results(
    results: list[dict],
) -> None:
    """
    벡터 검색의 두 번째 이후 결과를 표시한다.
    """

    if (
        len(results)
        <= 1
    ):
        return

    with st.expander(
        "추가 참고 문서 보기"
    ):
        for (
            index,
            result,
        ) in enumerate(
            results[1:4],
            start=1,
        ):
            st.markdown(
                f"### 참고 문서 {index}"
            )

            st.write(
                f"**문서:** "
                f"{result.get(
                    'source',
                    '알 수 없음',
                )}"
            )

            st.write(
                f"**문서 종류:** "
                f"{result.get(
                    'document_type',
                    '알 수 없음',
                )}"
            )

            st.write(
                f"**페이지:** "
                f"{result.get(
                    'page',
                    '알 수 없음',
                )}"
            )

            st.write(
                f"**청크 ID:** "
                f"{result.get(
                    'chunk_id',
                    '알 수 없음',
                )}"
            )

            st.write(
                f"**재정렬 점수:** "
                f"{result[
                    'final_score'
                ]:.4f}"
            )

            st.write(
                result["text"]
            )

            if (
                index
                < min(
                    len(results) - 1,
                    3,
                )
            ):
                st.divider()


# =========================================================
# Main
# =========================================================


def main() -> None:
    """
    Streamlit 화면의 전체 실행 순서.
    """

    # -----------------------------------------------------
    # UI 스타일
    # -----------------------------------------------------

    apply_custom_style()

    # -----------------------------------------------------
    # 화면 제목
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="main-title">
            AI 학사 문서 도우미
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-description">
            학교 공식 학사 문서와 교과과정 데이터를 기반으로
            관련 정보를 검색합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="system-description">
            졸업요건 · 2026학년도 교과과정(1~3학년) · AI·BigData 로드맵 · 대체과목 · 자격증 학점인정
            <br>장학금·등록금·행정 일정은 현재 지원하지 않습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # 검색 시스템 초기화
    # -----------------------------------------------------

    try:
        with st.spinner(
            "AI 모델과 문서 데이터베이스를 "
            "불러오는 중입니다..."
        ):
            (
                model,
                collection,
                intent_model,
            ) = (
                initialize_search_system()
            )

    except Exception as error:
        logging.exception("Search system initialization failed")
        st.error("검색 준비를 완료하지 못했습니다. 실행 담당자는 모델 파일과 문서 DB를 확인한 뒤 앱을 다시 실행해 주세요.")
        st.caption("실행 안내: README의 로컬 실행 항목. 상세 원인은 실행 터미널 로그에서 확인할 수 있습니다.")

        st.stop()

    # -----------------------------------------------------
    # DB 연결 상태
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="database-status">
            <span class="database-status-dot">●</span>
            검색 데이터 {collection.count()}개 연결됨
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # 질문 입력
    # -----------------------------------------------------

    st.caption("질문에 학년·학기 또는 과목명을 넣어 주세요. 졸업요건은 입학 유형과 2년제·3년제 여부를 함께 적으면 더 정확합니다.")
    with st.expander("질문 예시와 자료 범위"):
        st.markdown("- 3학년 여름계절학기 과목 알려줘\n- 신입학 3년제 전공 최소학점은?\n- AI 에이전트 프로그래밍은 로드맵에서 몇 학기에 있어?\n- AI프로그래밍 대체과목이 뭐야?\n- 대체수강 최대 성적은?\n- 정보처리산업기사는 몇 학점 인정돼?\n- 자격증 학점인정 신청 조건은?")
        st.caption("로드맵에는 4학년 위치도 있지만, 교과과정 데이터는 1~3학년만 있습니다. 졸업요건의 입학년도별 예외는 원문을 함께 확인해 주세요.")
        st.caption("교과과정·대체과목·자격증 매칭 표는 2026학년도 AI소프트웨어학과 자료입니다. 새 안내문은 2026학년도 2학기 기준이며, 원문 학과명 '컴퓨터정보공학과'도 함께 검색합니다. 다른 학과·다른 연도로 확대 적용하지 않습니다.")
    with st.form("academic_search"):
        question = st.text_input("질문을 입력하세요.", placeholder="예: 3학년 2학기 전공 과목 알려줘")
        search_button = st.form_submit_button("검색", type="primary", use_container_width=True)

    if not search_button:
        return

    question = (
        question.strip()
    )

    if not question:
        st.warning(
            "질문을 먼저 입력해 주세요."
        )

        return

    # -----------------------------------------------------
    # 검색 실행
    # -----------------------------------------------------

    try:
        with st.spinner(
            "학사 정보를 검색하고 있습니다..."
        ):
            # =================================================
            # 0. 공식 교과과정 미지원 범위
            # =================================================

            unsupported_message = (
                create_unsupported_curriculum_message(
                    question
                )
            )

            if (
                unsupported_message
                is not None
            ):
                st.subheader(
                    "답변"
                )

                st.write(
                    unsupported_message
                )

                return

            # =================================================
            # 1. 직접 학습 모델 + 안전망으로 의도 판단
            # =================================================

            decision = (
                resolve_question_intent(
                    question=question,
                    collection=collection,
                    intent_model=(
                        intent_model
                    ),
                )
            )

            final_intent = (
                decision[
                    "final_intent"
                ]
            )

            question = retrieval_question_for_decision(question, decision, collection)

            if decision.get("response_message"):
                st.subheader("답변")
                st.write(decision["response_message"])
                display_intent_decision(decision)
                return

            # =================================================
            # 2. 범위 밖 질문
            # =================================================

            if (
                final_intent
                == "out_of_scope"
            ):
                st.subheader(
                    "답변"
                )

                st.write(
                    out_of_scope_answer(question)
                )

                display_intent_decision(
                    decision
                )

                return

            # =================================================
            # 3. 교육과정 구조화 검색
            # =================================================

            if final_intent in NEW_INTENTS:
                extended = answer_extension(question, final_intent, collection, model)
                display_expansion_result(extended)
                display_intent_decision(decision)
                return

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

                if curriculum_results:
                    display_curriculum_results(
                        question=question,
                        results=(
                            curriculum_results
                        ),
                    )

                else:
                    st.subheader(
                        "답변"
                    )

                    st.write(
                        create_curriculum_answer(
                            question=question,
                            results=[],
                        )
                    )

                display_intent_decision(
                    decision
                )

                return

            # =================================================
            # 4. 졸업요건 / 로드맵 벡터 검색
            # =================================================

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

        # -----------------------------------------------------
        # 벡터 검색 결과 표시
        # -----------------------------------------------------

        if not results:
            st.warning(
                "관련 문서를 찾지 못했습니다."
            )

            display_intent_decision(
                decision
            )

            return

        if results[0]["final_score"] < MIN_FINAL_SCORE:
            st.warning(create_answer(question, results))
            display_intent_decision(decision)
            return

        display_vector_best_result(
            question=question,
            result=results[0],
        )

        display_additional_results(
            results
        )

        display_intent_decision(
            decision
        )

    except Exception as error:
        logging.exception("Academic search failed")
        st.error("검색을 완료하지 못했습니다. 잠시 후 다시 검색해 주세요. 같은 문제가 계속되면 실행 담당자에게 알려 주세요.")


def display_expansion_result(result: dict) -> None:
    """Display exact values and page-level evidence from the supplied PDFs."""
    st.subheader("답변")
    if result['status'] == 'answered':
        st.write(result.get('display_answer', result['answer']))
    else:
        st.info(result['answer'])
    if result['records'] and result['intent'] == 'substitute_course':
        st.dataframe([{
            '기존 과목': r['original_course'], '기존 코드': r['original_code'],
            '대체 과목': r['replacement_course'], '대체 코드': r['replacement_code'],
            '원문 관계': r['relation'], '원문 쪽': r['page'],
        } for r in result['records']], hide_index=True, use_container_width=True)
    if result.get('details'):
        with st.expander("상세 정보 보기"):
            st.write(result['details'])
    with st.expander("개발 정보 · 검색 근거"):
        st.json({key: result[key] for key in ['retrieval_mode', 'raw_ids', 'ranked_ids']})
        if result['records'] and result['intent'] == 'substitute_course':
            st.json([{'record_id': r['record_id'], '표 행': r['row']} for r in result['records']])
    sources = {s['source']:s for s in load_data('sources.json')}
    grouped = {}
    for evidence in result['evidence']:
        grouped.setdefault(evidence['source'], []).append(evidence)
    for source, evidence in grouped.items():
        pages = ', '.join(str(p) for p in sorted({e['page'] for e in evidence}))
        st.caption(f"근거: {source} · {pages}쪽 · 2026학년도")
        with st.expander(f"근거 원문 · {source}"):
            st.write(f"**PDF 쪽:** {pages} / 전체 {sources[source]['pages']}쪽")
            st.write("**원문 학과명:** " + ', '.join(sorted({r.get('department_original', '학교 공통') for r in evidence})))
            if sources[source]['extraction'] == 'manual_visual_transcription':
                st.caption("이미지 PDF의 해당 쪽을 직접 확인해 입력한 내용입니다. 원본 PDF를 함께 확인할 수 있습니다.")
            for item in evidence:
                if 'section' in item:
                    st.write(f"**{item['section']}**")
                    st.write(item['text'])
            pdf_path = Path(__file__).resolve().parent / sources[source]['relative_path']
            if pdf_path.is_file():
                st.download_button("원본 PDF 내려받기", data=pdf_path.read_bytes(), file_name=source,
                    mime='application/pdf', key='source_' + sources[source]['source_id'])
            else:
                st.warning("원본 PDF 파일이 없어 내려받기를 제공할 수 없습니다.")


if __name__ == "__main__":
    main()
