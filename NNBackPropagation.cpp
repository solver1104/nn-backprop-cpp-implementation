//#pragma warning( disable : 4996 )
#include <bits/stdc++.h>

//#pragma GCC target ("avx2")
//#pragma GCC optimization ("O3")
//#pragma GCC optimization ("unroll-loops")

using namespace std;

const int N1 = 784;
const int N2 = 400;
const int N3 = 100;
const int N4 = 10;

const int trainingCases = 9900;
const int testCases = 100;
const int batch = 110;
const int epochs = 50;
//const int learningRate = 2 * 0.01 / (N4 * batch);
const double learningRate = 0.01;

double V1[N1];
double V2[N2];
double V3[N3];
double V4[N4];
double EL1[N1][N2];
double EL2[N2][N3];
double EL3[N3][N4];
double BL2[N2];
double BL3[N3];
double BL4[N4];

double gradientEL1[N1][N2];
double gradientEL2[N2][N3];
double tempSum[N2][N3];
double gradientEL3[N3][N4];
double gradientBL2[N2];
double tempGradient1[N2];
double gradientBL3[N3];
double tempGradient[N3];
double gradientBL4[N4];

double trData[trainingCases][N1];
int trDataAns[trainingCases];
double teData[testCases][N1];
int teDataAns[testCases];

pair<double, int> tempmax;

int shuffleTrain[trainingCases];

double a(double input) {
	return 1/(1 + exp(-input));
};

double da(double input) {
	return input * (1 - input);
};

int main()
{
	string problemName = "MNISTData";
	ifstream cin(problemName + ".in");
	ofstream cout(problemName + ".out");
	
	ios::sync_with_stdio(0);
	cin.tie(0);

	default_random_engine generator;
	uniform_real_distribution<double> distribution(-1.0, 1.0);
	mt19937 g(chrono::steady_clock::now().time_since_epoch().count());
	auto begin = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < trainingCases; i++) {
		shuffleTrain[i] = i;
	}

	for (int i = 0; i < trainingCases; i++) {
		for (int j = 0; j < N1; j++) {
			cin >> trData[i][j];
			trData[i][j] = 0.99 * trData[i][j] / 255 + 0.01;
		}
		cin >> trDataAns[i];
	}

	for (int i = 0; i < testCases; i++) {
		for (int j = 0; j < N1; j++) {
			cin >> teData[i][j];
			teData[i][j] = 0.99 * teData[i][j] / 255 + 0.01;
		}
		cin >> teDataAns[i];
	}

	for (int i = 0; i < N1; i++) {
		for (int j = 0; j < N2; j++) {
			EL1[i][j] = distribution(generator);
		}
	}

	for (int i = 0; i < N2; i++) {
		for (int j = 0; j < N3; j++) {
			EL2[i][j] = distribution(generator);
		}
	}

	for (int i = 0; i < N3; i++) {
		for (int j = 0; j < N4; j++) {
			EL3[i][j] = distribution(generator);
		}
	}

	for (int i = 0; i < N2; i++) {
		BL2[i] = distribution(generator);
	}

	for (int i = 0; i < N3; i++) {
		BL3[i] = distribution(generator);
	}

	for (int i = 0; i < N4; i++) {
		BL4[i] = distribution(generator);
	}

	for (int epoch = 0; epoch < epochs; epoch++) {
		shuffle(shuffleTrain, shuffleTrain + trainingCases, g);
		for (int i = 0; i < trainingCases / batch; i++) {
			for (int j = 0; j < batch; j++) {
				for (int k = 0; k < N1; k++) {
					V1[k] = trData[shuffleTrain[i * batch + j]][k];
					for (int l = 0; l < N2; l++) {
						V2[l] += V1[k] * EL1[k][l];
					}
				}

				for (int k = 0; k < N2; k++) {
					V2[k] += BL2[k];
					V2[k] = a(V2[k]);

					for (int l = 0; l < N3; l++) {
						V3[l] += V2[k] * EL2[k][l];
					}
				}

				for (int k = 0; k < N3; k++) {
					V3[k] += BL3[k];
					V3[k] = a(V3[k]);

					for (int l = 0; l < N4; l++) {
						V4[l] += V3[k] * EL3[k][l];
					}
				}

				for (int k = 0; k < N4; k++) {
					V4[k] += BL4[k];
					V4[k] = a(V4[k]);
				}

				for (int k = 0; k < N4; k++) {
					gradientBL4[k] += (V4[k] - ((k) == trDataAns[shuffleTrain[i * batch + j]])) * V4[k] * (1 - V4[k]);
				}

				for (int k = 0; k < N3; k++) {
					for (int l = 0; l < N4; l++) {
						gradientEL3[k][l] += (V4[l] - ((l) == trDataAns[shuffleTrain[i * batch + j]])) * V4[l] * (1 - V4[l]) * V3[k];
					}
				}

				for (int k = 0; k < N3; k++) {
					tempGradient[k] = 0;
					for (int l = 0; l < N4; l++) {
						gradientBL3[k] += (V4[l] - ((l) == trDataAns[shuffleTrain[i * batch + j]])) * V4[l] * (1 - V4[l]) * EL3[k][l] * V3[k] * (1 - V3[k]);
						tempGradient[k] += (V4[l] - ((l) == trDataAns[shuffleTrain[i * batch + j]])) * V4[l] * (1 - V4[l]) * EL3[k][l] * V3[k] * (1 - V3[k]);
					}
				}

				for (int k = 0; k < N2; k++) {
					for (int l = 0; l < N3; l++) {
						gradientEL2[k][l] += tempGradient[l] * V2[k];
						tempSum[k][l] = tempGradient[l] * EL2[k][l];
					}
				}

				for (int k = 0; k < N2; k++) {
					tempGradient1[k] = 0;
					for (int l = 0; l < N3; l++) {
						gradientBL2[k] += tempSum[k][l] * V2[k] * (1 - V2[k]);
						tempGradient1[k] += tempSum[k][l] * V2[k] * (1 - V2[k]);
					}
				}

				for (int k = 0; k < N1; k++) {
					for (int l = 0; l < N2; l++) {
						gradientEL1[k][l] += tempGradient1[l] * V1[k];
					}
				}

				fill(V2, V2 + N2, 0);
				fill(V3, V3 + N3, 0);
				fill(V4, V4 + N4, 0);
			}

			for (int j = 0; j < N1; j++) {
				for (int k = 0; k < N2; k++) {
					EL1[j][k] = EL1[j][k] - learningRate * gradientEL1[j][k];
					gradientEL1[j][k] = 0;
				}
			}

			for (int j = 0; j < N2; j++) {
				for (int k = 0; k < N3; k++) {
					EL2[j][k] = EL2[j][k] - learningRate * gradientEL2[j][k];
					gradientEL2[j][k] = 0;
				}
			}

			for (int j = 0; j < N3; j++) {
				for (int k = 0; k < N4; k++) {
					EL3[j][k] = EL3[j][k] - learningRate * gradientEL3[j][k];
					gradientEL3[j][k] = 0;
				}
			}

			for (int j = 0; j < N2; j++) {
				BL2[j] = BL2[j] - learningRate * gradientBL2[j];
				gradientBL2[j] = 0;
			}

			for (int j = 0; j < N3; j++) {
				BL3[j] = BL3[j] - learningRate * gradientBL3[j];
				gradientBL3[j] = 0;
			}

			for (int j = 0; j < N4; j++) {
				BL4[j] = BL4[j] - learningRate * gradientBL4[j];
				gradientBL4[j] = 0;
			}
		}
	}

	for (int i = 0; i < testCases; i++) {
		for (int k = 0; k < N1; k++) {
			V1[k] = teData[i][k];
			for (int l = 0; l < N2; l++) {
				V2[l] += V1[k] * EL1[k][l];
			}
		}

		for (int k = 0; k < N2; k++) {
			V2[k] += BL2[k];
			V2[k] = a(V2[k]);

			for (int l = 0; l < N3; l++) {
				V3[l] += V2[k] * EL2[k][l];
			}
		}

		for (int k = 0; k < N3; k++) {
			V3[k] += BL3[k];
			V3[k] = a(V3[k]);

			for (int l = 0; l < N4; l++) {
				V4[l] += V3[k] * EL3[k][l];
			}
		}

		tempmax = { 0.0,0 };

		for (int k = 0; k < N4; k++) {
			V4[k] += BL4[k];
			V4[k] = a(V4[k]);
			tempmax = max(tempmax, { V4[k], k });
		}

		cout << tempmax.first << " " << tempmax.second << " " << teDataAns[i] << "\n";
	}

	cout << "\n";

	for (int i = 0; i < N1; i++) {
		for (int j = 0; j < N2; j++) {
			cout << EL1[i][j] << " ";
		}
	}
	
	cout << "\n";

	for (int i = 0; i < N2; i++) {
		for (int j = 0; j < N3; j++) {
			cout << EL2[i][j] << " ";
		}
	}

	cout << "\n";

	for (int i = 0; i < N3; i++) {
		for (int j = 0; j < N4; j++) {
			cout << EL3[i][j] << " ";
		}
	}

	cout << "\n";

	for (int i = 0; i < N2; i++) {
		cout << BL2[i] << " ";
	}

	cout << "\n";

	for (int i = 0; i < N3; i++) {
		cout << BL3[i] << " ";
	}

	cout << "\n";

	for (int i = 0; i < N4; i++) {
		cout << BL4[i] << " ";
	}

	cout << "\n";

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

	cout << "Training elapsed time: " << setprecision(5) << elapsed.count() * 1e-9 << " seconds";
}