"""2026 extension: exact table relationships + source-grounded E5 rule retrieval.

No generation, fuzzy qualification equivalence, or transitive course replacement.
The returned trace is shared by UI and evaluation.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data/data_expansion_v3'
NEW_INTENTS = {'substitute_course': '대체과목', 'certificate_credit': '자격증학점인정'}

@lru_cache(maxsize=None)
def load_data(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))

def compact(text):
    # Preserve +/#/dots: C, C++, C# and ASP.NET are distinct course names.
    return re.sub(r'\s+', '', str(text)).lower()

def explicit_extension_intent(question):
    from request_context import get_request_context
    context = get_request_context(str(question))
    if context['request_intent'] in NEW_INTENTS:
        return context['request_intent']
    if context['request_intent'] in {'graduation', 'curriculum', 'roadmap', 'out_of_scope'}:
        return None
    question = context['request_text']
    q = compact(question)
    # Inspect the requested side of a contrast; never rewrite course/entity names.
    requested = re.split(r'말고|아니고|아니라', q)[-1]
    if requested != q and re.search(r'학점|학기|학년|이수구분', requested) and not any(
            w in requested for w in ['인정', '매칭', '대체', '재수강', '자격', '처리', '신청']):
        return None
    # Mentioning a table is not a request for its recognition/replacement rules.
    focus = re.search(r'(?:과목|수업)자체(?:의|에배정된)?(?:학점|이수구분)|정규수업.*?(?:학기|학년|개설)', q)
    if focus and not any(w in q[focus.end():] for w in ['인정', '매칭', '대체', '자격', '받', '처리', '신청']):
        return None
    cert = (any(compact(r['certificate']) in q for r in load_data('certificate_records.json'))
            or any(w in q for w in ['자격증', '정보처리산업기사', '정보처리기사', '정처산기', 'sqld', '컴활'])
            or bool(re.search(r'산업기사|기사|기능사|능력[12]급', q)))
    # A certificate entity alone (or "전공/수강/대체" alone) is not credit recognition.
    credit = any(w in q for w in ['학점', '인정', '매칭', '이수', '평점'])
    if (cert and credit) or '매칭교과목' in q or '매칭과목' in q:
        return 'certificate_credit'
    if any(w in q for w in ['대체', '재수강']):
        return 'substitute_course'
    if '대신' in q:
        names = [r[key] for r in load_data('substitute_records.json')
                 for key in ['original_course', 'replacement_course']]
        course_context = any(compact(name) in q for name in names) or any(w in q for w in ['과목', '교과목', '수업'])
        relation = bool(re.search(r'듣|(?<!만)(?:들어|들을|들으면)|수강|이수|인정|대신.*?(?:과목|교과목|수업)', q))
        if course_context and relation:
            return 'substitute_course'
    return None

def scope_error(question):
    q = compact(question)
    aliases = load_data('department_aliases.json')['aliases']
    departments = re.findall(r'[가-힣A-Za-z]+학과', question)
    if any(d not in aliases + ['우리학과', '해당학과', '소속학과'] for d in departments):
        return '현재 확장 자료는 AI소프트웨어학과(원문: 컴퓨터정보공학과)만 지원합니다. 다른 학과의 대체과목·자격증 인정 기준은 확인할 수 없습니다.'
    year = re.search(r'(20\d{2})\s*학년도', question)
    if not year:
        year = re.search(r'(20\d{2})\s*년\s*(?:기준|설정표|대체과목|학점인정)', question)
    if year and year[1] != '2026':
        return f"{year[1]}학년도 확장 자료는 없습니다. 현재 자료는 2026학년도 기준이므로 해당 연도에 적용할 수 있는지 확인할 수 없습니다."
    if '2026' in q and any(w in q for w in ['1학기신청', '1학기자격증신청']):
        return '신청 안내는 2026학년도 2학기 자료입니다. 1학기 신청 조건이나 일정은 확인할 수 없습니다.'
    return None

def _longest_names(question, names):
    q = compact(question)
    selected = []
    for name in sorted(set(names), key=lambda n: len(compact(n)), reverse=True):
        n = compact(name)
        if n in q and not any(n in compact(prior) for prior in selected):
            selected.append(name)
    return selected

def substitute_lookup(question):
    rows = load_data('substitute_records.json')
    q = compact(question)
    reverse = any(w in q for w in ['으로대체되는원래', '으로대체가능한기존', '으로대체할수있는예전', '으로바뀐과목', '의원래과목'])
    field = 'replacement_course' if reverse else 'original_course'
    all_names = [r[f] for r in rows for f in ['original_course', 'replacement_course']]
    names = [n for n in _longest_names(question, all_names) if n in {r[field] for r in rows}]
    # Codes are exact and retain their direction. Names take priority.
    if names:
        matches = [r for r in rows if r[field] in names]
    else:
        code_field = 'replacement_code' if reverse else 'original_code'
        codes = {r[code_field] for r in rows if re.search(r'(?<![A-Za-z0-9])' + re.escape(r[code_field]) + r'(?![A-Za-z0-9])', question, re.I)}
        matches = [r for r in rows if r[code_field] in codes]
    # Tables answer only recorded edges, not chains. Include all alternatives.
    return matches, reverse

def rule_search(question, intent, collection, model, k=6):
    from step6_retrieval_answer import create_query_embedding
    data = collection.query(query_embeddings=[create_query_embedding(model, question)], n_results=k,
        where={'$and': [{'domain': NEW_INTENTS[intent]}, {'record_kind': 'rule'}]},
        include=['documents', 'metadatas', 'distances'])
    rules = {r['id']: r for r in load_data('rules.json')}
    raw = []
    for meta, distance in zip(data['metadatas'][0], data['distances'][0]):
        rule = rules[str(meta['chunk_id'])]
        # Small, inspectable lexical boost within E5's top-K; no extra model.
        q = compact(question)
        hits = [kw for kw in rule['keywords'] if len(compact(kw)) >= 2 and compact(kw) in q]
        score = 1 - float(distance)
        raw.append({**rule, 'semantic_similarity': score,
                    'keyword_hits': hits, 'final_score': score + min(len(hits), 3) * .08})
    ranked = sorted(raw, key=lambda r: r['final_score'], reverse=True)
    return raw, ranked

def answer_extension(question, intent, collection=None, model=None):
    if intent not in NEW_INTENTS:
        raise ValueError('Not an extension intent')
    result = dict(intent=intent, answer='', status='answered', records=[], evidence=[],
                  raw_ids=[], ranked_ids=[], retrieval_mode='structured')
    err = scope_error(question)
    if err:
        result.update(answer=err, status='unsupported_scope')
        return result
    q = compact(question)
    header = '2026학년도 AI소프트웨어학과 기준입니다.\n'
    from question_policy import input_policy
    policy = input_policy(question)
    if policy and policy.get('response_message'):
        result.update(answer=policy['response_message'], status=policy['response_kind'])
        return result
    if intent == 'substitute_course':
        records, reverse = substitute_lookup(question)
        rule_request = any(w in q for w in ['조건','성적','최대','최고','b+','a+','a0','학점처리','증명서','평점','기간','언제','날짜','신청방법','차이','잔여석','계열','위탁','삭제','정원','폐강','합반','수강구분'])
        if records and not rule_request:
            result['records'] = records
            result['evidence'] = records
            result['ranked_ids'] = [r['record_id'] for r in records]
            lines = [header, '설정표에서 확인된 관계입니다.' if not reverse else '역방향 조회 결과이며, 아래 화살표는 원문의 방향입니다.']
            for r in records:
                lines.append(f"- {r['original_course']} ({r['original_code']}) → {r['replacement_course']} ({r['replacement_code']}) · {r['relation']}")
            lines.append("표에 있는 직접 관계만 표시했습니다. '동일*'와 '동일(대체)'는 원문 표기이며, 이번 학기 개설·잔여석 및 신청 조건은 별도 확인해야 합니다.")
            result['answer'] = '\n'.join(lines)
            return result
        if not rule_request and not any(w in q for w in ['대체수강', '재수강', '대체과목안내', '대체과목이뭐', '대체과목설명']):
            result.update(status='not_found', answer=header + '해당 과목의 대체 관계를 대상 학과의 2026학년도 설정표에서 확인할 수 없습니다. 과목명 또는 과목코드를 정확히 입력해 주세요. 비슷한 과목이나 다른 학과의 관계를 대신 안내하지 않습니다.')
            return result
    else:
        certs = load_data('certificate_records.json')
        known = certs[0]
        names = re.findall(r'[가-힣A-Za-z0-9]+(?:산업기사|기사|기능사|능력[12]급)', question)
        unsupported = [n for n in names if n != known['certificate']]
        unsupported += [w for w in ['sqld', 'sqlp', '컴활', '컴퓨터활용능력', '빅데이터분석기사'] if w in q]
        if unsupported:
            result.update(status='not_found', answer=header + '질문한 자격증의 학점인정 근거를 대상 학과 자료에서 확인할 수 없습니다. 현재 확인된 자격증은 정보처리산업기사(2학점)입니다. 정보처리기사 등 다른 자격증이나 타 학과 기준으로 대체 판단하지 않습니다.')
            return result
        asks_rules = any(w in q for w in ['조건','대상','신청','입학','휴학','재학','이미','모두','듣고','수강중','이수한','끝냈','졸업','평점','평균','gpa','성적','등급','여러','두개','몇개','횟수','다르','교양','전공심화','최소','최대'])
        if not asks_rules:
            result['records'] = certs
            result['evidence'] = certs
            result['ranked_ids'] = [known['record_id']]
            result['answer'] = header + f"정보처리산업기사의 인정학점은 {known['credits']}학점입니다.\n매칭 교과목: " + ', '.join(known['matched_courses']) + '.\n10개 매칭 과목 각각에 2학점을 주거나 20학점을 합산하는 의미가 아닙니다. 신청 조건과 기존 수강·이수 상태를 함께 확인해야 합니다.'
            # Keep the existing detailed return contract for callers/tests.
            # The user-facing answer shows only the requested item.
            result['details'] = result['answer']
            asks_matches = any(w in q for w in ['매칭', '연결', '목록', '과목'])
            requested_names = _longest_names(question, known['matched_courses'])
            if asks_matches and requested_names:
                result['display_answer'] = header + ', '.join(requested_names) + '은(는) ' + known['certificate'] + '의 매칭 교과목에 포함됩니다.'
            elif asks_matches:
                result['display_answer'] = header + '매칭 교과목: ' + ', '.join(known['matched_courses']) + '.'
            else:
                result['display_answer'] = f"{known['certificate']}의 인정학점은 {known['credits']}학점입니다.\n(2026학년도 AI소프트웨어학과 기준)"
            return result
    if collection is None or model is None:
        raise ValueError('Rule retrieval requires the E5 model and expanded collection')
    raw, ranked = rule_search(question, intent, collection, model)
    result.update(retrieval_mode='vector_rules', raw_ids=[r['id'] for r in raw], ranked_ids=[r['id'] for r in ranked])
    if not ranked or ranked[0]['semantic_similarity'] < .5:
        result.update(status='not_found', answer=header + '관련 조건을 원문에서 충분히 확인할 수 없습니다. 질문을 구체적으로 입력해 주세요.')
        return result
    # First source paragraph is the answer; nearby evidence stays visible separately.
    selected = ranked[:2] if len(ranked) > 1 and ranked[1]['keyword_hits'] else ranked[:1]
    result['evidence'] = selected
    result['answer'] = header + '\n\n'.join(r['text'] for r in selected)
    if intent == 'substitute_course' and ranked[0]['id'] == 'sub_grade' and any(w in q for w in ['최대', '최고', '상한']):
        grade = re.search(r'최대\s*([A-F][+0-]?)', ranked[0]['text'])
        if grade:
            result['details'] = result['answer']
            result['display_answer'] = f"대체수강으로 취득할 수 있는 최대 성적은 {grade.group(1)}입니다.\n(2026학년도 기준)"
    return result
