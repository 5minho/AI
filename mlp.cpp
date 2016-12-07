#include <iostream>
#include <cmath>

void print(double(*w)[3],int row, int col) {
    for(int i = 0 ; i < row ; ++i) {
        for(int j = 0 ; j < col ; ++j)
            std::cout << w[i][j] << ' ';
        std::cout << '\n';
    }
    std::cout<<'\n';
}

int main() {
    const int N = 2, // 두 개의 입력 뉴런
        H = 2, // 두 개의 은닉 뉴런
        M = 1; // 한 개의 출력 뉴런
    int count = 0; // 반복학습 횟수 계수기의 초기화
    
    double eta = 0.2, // 학습률 설정
           E = 100.0, // while로 진입하기 위해 비용함수 큰 값으로 초기화
           Es = 0.01; // 프로그램을 정지시킬 때 기준이 되는 비용 함수

    double w_ji[N][N+1] = {
        {0.45, 0.11, 0.39},
        {-0.27, -0.01, 0.26}
    };// 입력층 -> 은닉층 가중치 설정

    double w_kj[M][H+1] = {
        {-0.04, -0.48, 0.32}
    };// 은닉층 -> 출력층 가중치 설정

      
    double net_pj[4][H+1] = {0.0, }; // 은닉 노드 입력 값
    double o_pj[4][H+1] = {0.0, }; // 은닉 노드 출력 값

    double net_pk[4][M] = {0.0, }; // 출력 노드 입력 값
    double o_pk[4][M] = {0.0, }; // 출력 노드 출력 값
   
    double delta_pk[4][M] = {0.0, }; //출력 노드 델타 값
    double delta_pj[4][H] = {0.0, }; // 은닉 노드 델타 값


    double Error[10000] = {0.0,}; //에러 기록

    int p = 0,
        j = 0,
        i = 0,
        k = 0;

    while(E > Es) {
        ++count;
        double Xp[4][N+1] = {
             {0.0, 0.0, -1.0},
             {0.0, 1.0, -1.0},
             {1.0, 0.0, -1.0},
             {1.0, 1.0, -1.0}
        };// 입력 값

        double Dp[4] = {0.0, 1.0, 1.0, 0.0}; // 출력 값
        double E_p[4] = {0.0, }; // 패턴별 비용함수
        E = 0.0;// 비용 함수 0 으로 초기화 
       
        for(p = 0 ; p < 4 ; ++p) {
            for(j = 0 ; j < H ; ++j) {
                net_pj[p][j] = 0.0;
                for(i = 0 ; i < N + 1 ; ++i)
                    net_pj[p][j] += (w_ji[j][i] * Xp[p][i]); // F-2 단계 : 은닉 뉴론의 입력 합
                o_pj[p][j] = 1.0 / (1.0 + exp(-net_pj[p][j])); // F-3 단계 : 은닉 뉴론의 출력
            }

            o_pj[p][j] = -1.0; // 출력 노드의 임계치를 배우기 위한 설정
        
            for(k = 0 ; k < M ; ++k) {
                net_pk[p][k] = 0.0;
                for(j = 0 ; j < H + 1 ; ++j)
                    net_pk[p][k] += (w_kj[k][j] * o_pj[p][j]); // F-4 단계 : 출력 뉴론의 입력 합
                o_pk[p][k] = 1.0 / (1.0 + exp(-net_pk[p][k])); // F-5 단계 : 출력 뉴론의 실제 출력 값
                //std::cout << "p가 " << p << " 일때 ";
                //std::cout << "출력 뉴론의 실제 출력 값 : " << o_pk[p][k] << std::endl; 
            }

            E_p[p] = 0.0; // 데이터 패턴별 비용함수 초기화
            
            for(k = 0 ; k < M ; ++k) { // B-1 단계 : 출력 뉴론의 델타 계산
                delta_pk[p][k] = (Dp[p] - o_pk[p][k]) * o_pk[p][k] * (1.0 - o_pk[p][k]);
                E_p[p] += (0.5 * pow((Dp[p] - o_pk[p][k]), 2.0));
            }

            E += E_p[p]; // 누적 비용 함수

            for(j = 0 ; j < H ; ++j) { // B-2 단계 : 은닉 뉴론의 델타 계산
                delta_pj[p][j] = 0.0;
                for(k = 0 ; k < M ; ++k) 
                  delta_pj[p][j] += (delta_pk[p][k] * w_kj[k][j] * o_pj[p][j] * (1.0 - o_pj[p][j]));
            }

            for(k = 0 ; k < M ; ++k) // B-3 단계 : 은닉층과 출력층간의 연결 가중치 갱신
                for(j = 0 ; j < H + 1; ++j)
                    w_kj[k][j] += (eta * delta_pk[p][k] * o_pj[p][j]);
        
            for(j = 0 ; j < H ; ++j) // B-4 단계 : 은닉층과 입력층간의 연결 가중치 갱신
                for(i = 0; i < N + 1; ++i)
                    w_ji[j][i] += (eta * delta_pj[p][j] * Xp[p][i]);
        }
        Error[count] = E;
        std::cout<< "cost : " << E << std::endl;
    }
    std::cout<<"epoch : " << count <<std::endl;
    for(p = 0 ; p < 4 ; ++p) {
        std::cout<< p << " : " << o_pk[p][0] <<std::endl; 
    }
    std::cout << "학습 완료 후 연결 가중치(출력층 -> 은닉층)" << std::endl;
    print(w_ji, N, N+1);
    std::cout << "학습 완료 후 연결 가중치(은닉층 -> 출력층)" << std::endl;
    print(w_kj, M, H+1);
}
