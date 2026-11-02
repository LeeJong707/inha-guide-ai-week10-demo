# AI 학사 문서 도우미 배포본

Python 3.12에서 `pip install -r requirements.txt` 후
`python -m streamlit run app.py`로 실행합니다.

Streamlit Community Cloud: 이 폴더의 파일을 저장소 루트로 사용하고,
entrypoint는 app.py, Python은 3.12로 설정합니다.
API 키나 .env는 필요하지 않습니다. 첫 실행에는 고정된 공개 E5 모델을
다운로드하므로 네트워크와 메모리가 필요합니다.
ChromaDB 폴더는 초기화 시 내부 기록을 쓸 수 있어야 합니다.

활성 모델: 8주차 300문항 Intent 모델. threshold 0.50.
공식 자료: 2026학년도 자료 등 PDF 6개 및 교과과정, 검색 단위 105개.
최신 학교 공지와의 차이는 별도 확인이 필요합니다.

Windows 검증과 Linux/Cloud 실서버 검증은 구분합니다.
실제 외부 배포 전 dependency 설치, DB count 105, 대표 답변/출처를 확인하세요.
