"""Bounded request-object evidence for existing six-way routing.

No model, corpus or query rewriting: select an original text span and describe
the requested information. Background mentions alone are not routing evidence.
"""
import re
from functools import lru_cache


def _compact(text):
    return re.sub(r'\s+', '', str(text)).lower()


def _signals(text):
    q = _compact(text)
    scores = {}
    evidence = {}

    def add(intent, score, reason):
        scores[intent] = score
        evidence[intent] = reason

    # Unsupported requests are objects/actions, not incidental background words.
    external = re.search(
        r'(?:졸업증명서|증명서발급|졸업식).*(?:발급|수수료|배송|날짜|언제|장소)|'
        r'(?:전화번호|연락처|이메일주소).*(?:알려|찾|필요|궁금|뭐)|'
        r'(?:시험|실기|필기).*(?:원서|수험표|응시료|접수사이트|시험장)|'
        r'(?:취업률|채용마감|채용공고|기출문제)|'
        r'(?:코드|로그인화면|블로그|페이지디자인).*(?:만들|작성|완성|짜|고쳐)|'
        r'(?:날씨|우산|기온|한파|막차|기숙사|학식|학생증|비밀번호)|'
        r'(?:장학금|등록금|도서관|셔틀버스|학사일정|중간고사|기말고사)|'
        r'(?<!대체)수강신청.*(?:날짜|시작|시간)|(?:휴학|복학)(?:신청|절차)|'
        r'(?:메일|이메일|메시지|카카오톡|자기소개).*(?:써줘|써주|작성)', q)
    form_entry = '신청서' in q and bool(re.search(r'(?:전화번호|연락처).*(?:적|기입|작성|써야)', q))
    if external and not form_entry:
        add('out_of_scope', 10, 'unsupported_requested_object')

    degree = re.search(r'졸업|학위|전문학사|[23삼이]년(?:제|과정)|학교.*마칠', q)
    amount = re.search(r'학점|이수량|최소|최저|하한|요건|기준|통과선|총량|합계|전공몫|교양몫', q)
    minimum = re.search(r'(?:전공|교양).*(?:최소|최저|하한|이수량|기준|요건|총량)|'
                        r'(?:최소|최저).*(?:전공|교양|이수학점)', q)
    if (degree and amount) or minimum:
        add('graduation', 5, 'degree_credit_requirement')

    qualification = re.search(r'자격(?:증|취득|학점|인정)|취득자격|정보처리(?:산업)?기사|'
                              r'정처산기|산업기사|sqld|컴활|매칭', q)
    recognition = re.search(r'학점|인정|매칭|이수|평점', q)
    if qualification and recognition:
        add('certificate_credit', 7, 'qualification_recognition_request')

    replacement = re.search(r'대체|재수강|재이수|갈음|갈아.*듣|바꿔.*이수|다시.*이수', q)
    historical = re.search(r'폐지|없어진|없어져|사라|옛|예전|미이수|미취득|못(?:땄|딴|이수|끝|듣)|낙제|np|f받|칸이비어|새이름', q)
    relation = re.search(r'대신.*(?:듣|들어|들을|수업|과목|교과목)|'
                         r'(?:채[우울워]|메[우울워]|연결|관계|설정표|갈음|갈아|바꿔|정리|다시처리|다시이수|재이수)', q)
    instead_relation = re.search(r'대신(?:할|하는|인정되는|인정될|가능한)?(?:정규)?(?:교과목|과목|수업|강의)|'
                                  r'(?:과목|교과목|수업|강의).{0,8}대신(?:듣|들어|인정|하)', q)
    if ((replacement and not re.search(r'대신.*(?:만들|작성|완성|써줘)', q))
            or (historical and relation) or instead_relation):
        add('substitute_course', 7, 'course_replacement_relation_or_rule')

    formal = re.search(r'과목|교과목|교과과정|교육과정|수업|강의|편성표|수업표|이수구분|전공이론|전공실습|'
                       r'프로그래밍|기초|설계|개론|실습|배정된|전공구분|이론선택|실습선택', q)
    attribute = re.search(r'학점|학년|학기|개설|편성|이수구분|전공구분|전공이론|전공실습|'
                          r'이론선택|실습선택|분류|시수|어느칸|과목코드', q)
    if formal and attribute:
        add('curriculum', 4, 'formal_course_attribute_or_list')
    specific_attribute = re.search(r'(?:정규|정식|강의|수업|과목).*?(?:학점|이수구분|편성|개설)|'
                                   r'(?:배정|자체)(?:된|의|에배정된)?학점', q)
    recognition_context = qualification and re.search(r'학점인정|인정신청|자격증으로|취득분으로|매칭', q)
    relation_context = replacement or (historical and relation) or instead_relation
    if (specific_attribute and not recognition_context and not relation_context
            and not re.search(r'인정|매칭|대체|자격|신청|이수처리|흐름|경로|로드맵|트랙', q[specific_attribute.end():])):
        add('curriculum', 8, 'specific_course_attribute_request')

    official_path = re.search(r'로드맵|트랙|진로(?:표|안내|그림|흐름|경로)|'
                              r'(?:학교|학과).*(?:안내|추천|제시|권|진로|길)|'
                              r'(?:ai|빅데이터|bigdata).*(?:한그림|진로|배움.*경로|공부.*경로)|'
                              r'(?:배우|공부).*(?:다음|이후).*(?:안내|제시)', q)
    learning = re.search(r'로드맵|트랙|공부|학습|배움|배우|단계|순서|흐름|경로|구간|이어|위치|배치|진로|학년별', q)
    if official_path and learning:
        add('roadmap', 6, 'official_learning_path')

    if not scores:
        return None, [], scores
    best = max(scores.values())
    winners = [intent for intent, score in scores.items() if score == best]
    if len(winners) != 1:
        return None, ['conflicting_request_evidence'], scores
    intent = winners[0]
    return intent, [evidence[intent]], scores


# Suffix selection always retains exact original spelling, including C++/#/ASP.NET.
_CLAUSE = re.compile(r'[。!?！？\n]+|(?<=[가-힣])\.(?:\s+|$)|,\s*')
_CONTRAST = re.compile(r'말고|아니고|아니라|보다는|보다|별개로|상관없이|무관하게|관계없이|'
                       r'인데|지만|그런데|빼고|됐고|나서|필요\s*없고|'
                       r'(?:생각|계획|관심)(?:은|이)?\s*없(?:고|어|어요)|'
                       r'대신(?!할|하|해|들|듣|수강|인정|가능)|'
                       r'(?:지금\s*(?:은|필요한\s*건|묻는\s*건)|질문은|우선)')
_BACKGROUND_END = re.compile(r'(?:이미|다)?(?:확인했|확인한|봤고|봤어|알겠|알고있|알아요)|'
                             r'필요없|필요\s*없|아니야|아니에요|아닙니다|뒤에볼|나중에')


@lru_cache(maxsize=512)
def get_request_context(question):
    original = str(question).strip()
    # Prefer a later explicit request, keeping earlier clauses when the ending
    # lacks its own domain evidence ("서류는 어디로 내나요?" needs context).
    candidates = [(0, original)]
    for match in _CLAUSE.finditer(original):
        candidates.append((match.end(), original[match.end():].strip()))
    for match in _CONTRAST.finditer(original):
        candidates.append((match.end(), original[match.end():].strip()))
    selected = (0, original)
    original_intent, original_evidence, original_scores = _signals(original)
    for start, candidate in sorted(candidates, reverse=True):
        if not candidate:
            continue
        intent, evidence, scores = _signals(candidate)
        if not intent:
            continue
        # A final acknowledgement/negation is not an affirmative request.
        if _BACKGROUND_END.search(candidate) and not re.search(r'궁금|알려|확인해|찾|필요|보고싶|물어|싶어|싶습|인가|나요|야[?？]?', _compact(candidate)):
            continue
        selected = (start, candidate)
        original_intent, original_evidence, original_scores = intent, evidence, scores
        break
    return {'request_text': selected[1], 'request_intent': original_intent,
            'request_evidence': original_evidence, 'request_scores': original_scores,
            'background_omitted': selected[0] > 0}


def request_text(question):
    return get_request_context(str(question))['request_text']
