"""Bounded topic-only clarification policy using existing supported entities."""
import re
from expansion_search import compact, load_data

TOPICS = (
 ('graduation', ('졸업','졸업요건'), '총 졸업학점 / 전공 최소학점 / 교양 최소학점 중 무엇이 궁금한가요?'),
 ('curriculum', ('과목','교과목','교과과정','교육과정'), '찾고 싶은 학년·학기 또는 과목명을 입력해주세요.'),
 ('roadmap', ('로드맵','학과로드맵'), '확인할 학년·학기 또는 과목명을 입력해주세요. 전체 로드맵이 필요하면 전체 로드맵이라고 입력해주세요.'),
 ('substitute_course', ('대체과목','대체수강'), '과목명이나 알고 싶은 항목(대체과목 / 신청 조건 / 최대 성적)을 입력해주세요.'),
 ('certificate_credit', ('자격증','자격증학점인정'), '자격증명이나 알고 싶은 항목(인정학점/매칭과목/신청조건)을 입력해주세요.'),
 ('graduation', ('교양',), '교양과 관련해 어떤 정보가 궁금한가요?\n교양 최소학점 또는 교양 과목을 확인할 수 있습니다.'),
 ('out_of_scope', ('ai','인공지능'), 'AI 관련해서 어떤 정보를 찾으시나요? 로드맵, 교과목, 특정 과목 정보를 확인할 수 있습니다.'),
)

# A bounded suffix grammar applies only after a registered complete topic.
_TOPIC_SUFFIX = r"(?:에대해(?:서)?|관련해서)?(?:좀)?(?:알려주세요|알려줘|보여주세요|보여줘|부탁해|궁금해)?"


def _policy_text(question):
    # Keep +, #, and internal punctuation in course/certificate names intact.
    return compact(question).rstrip('?？!！.。~～')


def _is_topic_only(text, topic):
    return re.fullmatch(re.escape(compact(topic)) + _TOPIC_SUFFIX, text) is not None


def input_policy(question):
    # This comparison text never replaces the original retrieval question.
    q = _policy_text(question)
    if any(_is_topic_only(q, name) for name in ('안녕', '안녕하세요', '반가워')):
        return dict(final_intent='out_of_scope', response_kind='greeting', response_message='안녕하세요! 궁금한 학사정보를 물어보세요.')
    for intent, names, message in TOPICS:
        if any(_is_topic_only(q, name) for name in names):
            return dict(final_intent=intent, response_kind='clarification', response_message=message)
    for record in load_data('certificate_records.json'):
        if _is_topic_only(q, record['certificate']):
            return dict(final_intent='certificate_credit', response_kind='clarification', response_message=f"{record['certificate']}의 어떤 항목이 궁금한가요? 인정학점/매칭과목/신청조건 중 선택해주세요.")
    if any(_is_topic_only(q, name) for name in ('교양학점', '졸업학점', '전공최소학점', '교양최소학점')):
        return dict(final_intent='graduation')
    return None
