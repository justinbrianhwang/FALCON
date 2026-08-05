# FALCON 실험 실행 가이드 (3번 스위트)

실행에 필요한 내용만 담았습니다. 세 파트(이질성 스트레스, 비용 측정, 추가 데이터셋)가
한 명령으로 순차 실행됩니다.

## 1. 환경 설치 (최초 1회)

conda(미니콘다 가능)가 설치되어 있어야 합니다.

```bash
conda env create -f environment.yml
conda activate falcon
pytest        # 전부 통과하면 환경 OK
```

## 2. 데이터 준비 (최초 1회)

아무 설정 없이 아래만 실행하면 `./data`에 자동 다운로드됩니다.

```bash
python scripts/prepare_data.py --datasets mnist,fmnist,svhn
```

완료되면 `./data/processed/<이름>.pkl`이 생기고, 실험 코드는 이 pkl만 읽습니다.

## 3. 실험 실행 (수 시간 — 자기 전에 걸어두는 것을 권장)

```bash
python experiments/run_coauthor3_suite.py
```

세 파트가 차례로 실행됩니다:

1. **E3 이질성 스트레스** — Dirichlet 알파 4단계에서 FALCON vs 베이스라인 비교
2. **E8 비용 측정** — 클라이언트 10/25/50/100 스케일에서 기록·개입 시간과 저장 용량
3. **FMNIST/SVHN 복제** — 두 데이터셋에서 스테이지 4종 localization

중간에 끊겼으면 같은 명령을 다시 실행해 주세요.

## 4. 결과 보내기

```bash
python scripts/collect_output.py
# -> tmp/Output_YYYY-MM-DD_HH-MM-SS.zip 생성됨
```

생성된 zip 파일을 보내주세요. `--full`은 따로 요청받은 경우에만 붙여주세요.

## 문제 생기면

- `pytest` 실패: `conda env remove -n falcon` 후 1번부터 재시도.
- 데이터 다운로드 실패: 네트워크 문제 — 해당 데이터셋만 `--datasets svhn`처럼 재실행
  (이미 받은 것은 건너뜀).
- 실험 오류로 중단: 같은 명령 재실행 후에도 반복되면, 에러 메시지 전체와 함께
  `collect_output.py`가 만든 zip을 보내주세요.
