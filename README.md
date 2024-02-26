![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=250&section=header&text=Level2-MovieRecommendation&desc=RecSys-02&fontSize=50&fontColor=FFFFFF&fontAlignY=40)
- [랩업 레포트](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-02/blob/main/docs/MovieRec_RecSys_WarpUpReport.pdf)

# 프로젝트 개요
**주제 : 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측**
## 프로젝트 소개

![image](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-02/assets/97018869/3578084a-ce30-44c3-9988-7718045d5be7)

- implicit feedback으로 전처리된 MovieLens 데이터 활용
- implicit feedback 기반의 sequential recommendation 시나리오를 바탕으로 사용자의 time-orderd sequence에서 일부 item이 누락된 상황을 상정
- 사용자의 순차적인 이력(timestamp)과 implicit feedback을 고려한다는 점에서, 추천시스템 입문 강의들에서 자주 소개되는 explicit feedback 기반 협업 필터링 문제와 차별화
- Task : 사용자가 “시청한” 영화 중 마지막 영화를 포함하여 일부를 제거 한 후, 이를 예측하고, 그 중 맞춘 개수를 평가 점수로서 사용한다.

##  학습목표
공유하고, 질문하고, 토론한다.<br>
모델 하나를 깊게 파고 이해한다. <br>
가망이 없다면 미련 없이 돌아선다.<br>
궁금하면 일단 시도 해본다.<br>
모든 판단은 근거에 기반한다.<br>

# 팀소개

네이버 부스트캠프 AI Tech 6기 Level 2 Recsys 2조 **R_AI_SE** 입니다.

<aside>
    
💡 **R_AI_SE의 의미**

Recsys AI Seeker(추천시스템 인공지능 탐구자)를 줄여서 R_AI_SE입니다~


## 우리 조의 장점
### 특징
- 24시간코딩 이후 4시간 취침의 열정맨덜
- 티키타카
- 꿀잼

</aside>

## 👋 R_AI_SE의 멤버를 소개합니다 👋

### 🦹‍팀원소개
| 팀원   | 역할 및 담당                      |
|--------|----------------------------------|
| [김수진](https://github.com/guridon) |  Baseline 기반 3SRec/SASRec/BERT4Rec 실험, EASE/VASP baseline 구현, Recbole Framework 코드 분석 및 inference |
| [김창영](https://github.com/ChangZero) | NBCF, ALS 구현, CAR모델 학습 템플릿 작성 및 실험 |
| [박승아](https://github.com/SeungahP) | RecBole_GNN 기반 그래프 모델 실험, 앙상블코드 작성 |
| [전민서](https://github.com/Minseojeonn) | RecBole 기반 모델 실험 및 베이스라인 코드 작성, Bert4Rec_Random_query 구현, Negative Sampling |
| [한대희](https://github.com/DAEHEE97) | Negative Sampling, DeepFM, AutoRec, MultiVAE  |
| [한예본](https://github.com/Yebonn-Han) | RecBole 기반 모델 실험, BERT4Rec, Negative Sampling, SASRec/BERT4Rec query 방식 수정 시도 |

 전체  : 데이터 EDA

### 👨‍👧‍👦 Team 협업
### 📝 Ground Rule
#### 팀 규칙
- 모더레이터 역할
  - 순서 : 매일 돌아 가면서
  - 역할 : 피어세션 시 소개하고 싶은 곡 선정
- 데일리 스크럼
    - 오늘 학습 계획 정하기
    - Github PR 올린 것 코드리뷰 진행
- 피어세션
    - 모더레이터가 가져 온 노래 나올 때 각자 스트레칭 하기
    - 강의에 나오는 논문 리뷰하기
    - 미션 파일 코드 분석 발표하기
    - Github PR 올린 것 코드리뷰 진행

- 노션을 활용한 일정관리
    - 실험 일지 작성을 통한 원활한 결과 공유 도모

#### 깃 사용 규칙
1. 커밋 메세지 컨벤션 유다시티 스타일
2. 이슈 기반 작업
3. 깃허브 칸반 보드를 활용한 일정 관리
4. 데일리 스크럼/피어세션때 PR코드 리뷰 후 병합

<br>

# 프로젝트 개발 환경 및 기술 스택
## ⚙️ 개발 환경
- OS: Linux-5.4.0-99-generic-x86_64-with-glibc2.31
- GPU: Tesla V100-SXM2-32GB * 6
- CPU cores: 8

## 🔧 기술 스택
![](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white)
![](https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=Jupyter&logoColor=white)
![](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=black)
![](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=Pandas&logoColor=white)
![](https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=white)


![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=footer&)
