#if 0
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
int mkl = 1;
#else
int mkl = 0;
#endif

#include <assert.h>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <eigen\Sparse>


#include "cucg.h"



typedef Eigen::Matrix<float, -1, -1> Mat;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> CSR;
typedef Eigen::SparseMatrix<float, Eigen::ColMajor> CSC;
typedef Eigen::Matrix<float, -1, 1> Vec;



void printVector(float* b, int N) {
	printf("b:\n");
	for (int i = 0; i < N; i++)
		printf("%8.4f ", b[i]);
	printf("\n\n");
}

void printVector(const Vec& b) {
	for (int i = 0; i < b.size(); i++)
		printf("%8.4f ", b[i]);
	printf("\n\n");
}




#pragma region --cpu path
struct CholeskeySolver
{	
	typedef Eigen::SimplicialLLT<CSC> Solver;

	CSC A;
	Vec b;
	Vec x;
	Solver solver;
	std::vector<float> value_h;
	std::vector<float> xtmp;

	void compute(const CSC& in) {
		solver.compute(in);

		x.resize(A.rows());
		b.resize(A.rows());
	}

	void analysis(const CSC& in)
	{
		
		value_h.resize(in.data().size());

		A.resize(in.rows(), in.cols());
		A.resizeNonZeros(in.data().size());


		printf("in data size: %d\n", in.data().size());
		printf("A non-zeros: %d\n", A.nonZeros());

		x.resize(A.rows());
		b.resize(A.rows());
		xtmp.resize(A.rows());

		solver.analyzePattern(A);

		if (solver.info() != Eigen::Success) {
			throw std::exception("CholeskeySolver::analysis(): failed");
		}
	}

	void factor(const CSC& in)
	{
#if 1
		printf("A.nonZeros: %d\n", A.nonZeros());
		for (int r = 0; r < A.size(); r++)
			A.valuePtr()[r] = value_h[r];
#endif

		solver.factorize(A);

		if (solver.info() != Eigen::Success) {
			throw std::exception("CholeskeySolver::factor(): failed");
		}
	}

	void solve(float* d_x, const float* d_b)
	{
		int rows = A.rows();
		for (int i = 0; i < A.rows(); i++)
			b[i] = d_b[i];

#if 0
		b = solver.permutationP() * b;
		solver.matrixL().solveInPlace(b);
		solver.matrixU().solveInPlace(b);
		x = solver.permutationPinv() * b;
#else
		x = solver.solve(b);
#endif

		for (int i = 0; i < A.rows(); i++)
			xtmp[i] = x[i];
	}

	// since A = L*L'
	// this functions solves for L*u = b
	void solveL(float* d_u, const float* d_b)
	{
		for (int i = 0; i < A.rows(); i++)
			b[i] = xtmp[i];

		b = solver.permutationP() * b;
		solver.matrixL().solveInPlace(b);

		for (int i = 0; i < A.rows(); i++)
			xtmp[i] = b[i];

	}

	// this functions solves for L'*x = u
	void solveLt(float* d_x, const float* d_u)
	{

		for (int i = 0; i < A.rows(); i++)
			b[i] = xtmp[i];

		solver.matrixU().solveInPlace(b);
		x = solver.permutationPinv() * b;

		for (int i = 0; i < A.rows(); i++)
			xtmp[i] = x[i];

	}

	static void dumpSparseMatrix(const CSC& A, const char* filename)
	{
		FILE* pFile = fopen(filename, "w");
		if (!pFile)
			throw std::exception("dumpSparseMatrix: create file failed!");
		for (int r = 0; r < A.outerSize(); r++)
		{
			int rs = A.outerIndexPtr()[r];
			int re = A.outerIndexPtr()[r + 1];
			for (int c = rs; c < re; c++)
				fprintf(pFile, "%d %d %ef\n", r, A.innerIndexPtr()[c], A.valuePtr()[c]);
		}
		fclose(pFile);
	}
};
#pragma endregion



using namespace std;
using namespace Eigen;

void testMatrixMulti() {
	MatrixXd a = MatrixXd::Random(1000, 1000);  // Ëæ»ú³õÊ¼»¯¾ØÕó
	MatrixXd b = MatrixXd::Random(1000, 1000);

	double t0_ = clock();
	MatrixXd c = a * b;
	double endd = clock();
	double thisTime = (double)(endd - t0_) / CLOCKS_PER_SEC;

	cout << thisTime << endl;
	system("PAUSE");
}

CSC readCSC(const char* file) {
	printf("read A from %s\n", file);
	FILE *fp = freopen(file, "r", stdin);
	assert(fp);

	int rows, cols, nnz;
	scanf("%d\n", &rows);
	scanf("%d\n", &cols);
	scanf("%d\n", &nnz);

	CSC A(rows, cols);

	while (1) {
		int i, j;
		float value;
		if (-1 == scanf("%d %d %ef\n", &i, &j, &value))
			break;
		A.insert(i, j) = value;
	}

	fflush(fp);
	fclose(fp);
	freopen("CON", "r", stdin);
	assert(A.nonZeros() == nnz);

	printf("A: %d x %d, nnz: %d, den: %.2f%%\n", A.rows(), A.cols(), A.nonZeros(), A.nonZeros() * 100.0 / (A.rows() * A.cols()));

	return A;
}

Vec readVec(const char* file) {
	printf("read b from %s\n", file);
	FILE *fp = freopen(file, "r", stdin);

	int rows;
	scanf("%d\n", &rows);
	Vec b;
	b.resize(rows);

	int i = 0;
	while (1) {
		float value;
		if (-1 == scanf("%ef\n", &value))
			break;
		b[i++] = value;
	}

	fflush(fp);
	fclose(fp);
	freopen("CON", "r", stdin);
	printf("b: %d\n", b.size());

	return b;
}

void readEquation(CSC& A, Vec& b, const char* Afile, const char* bfile) {
	printf("reading equation:\n");
	A = readCSC(Afile);
	b = readVec(bfile);
	assert(A.cols() == b.size());
	printf("\n\n\n");
}

void eigenValues(const CSC& A) {
	printf("analyze eigen values:\n");
	printf("A: %d x %d, nz: %d\n", A.rows(), A.cols(), A.nonZeros());
	Mat A_ = A.toDense();
	Eigen::EigenSolver<Mat> eigen_solver(A_);

	int poss = 0, negs = 0, smalls = 0;
	auto eis = eigen_solver.eigenvalues();
	std::vector<float> evs;
	for (int i = 0; i < eis.size(); i++) {
		if (eis(i).real() > 0) {
			poss++;
		}
		else if (eis(i).real() < 0) {
			negs++;
		}
		evs.push_back(eis(i).real());
		if (fabs(eis(i).real()) < 1.0)
			smalls++;
	}
	printf("pos %d, neg %d, zero: [ %d ], smalls: %d, total %d\n", poss, negs, 
		eis.size() - poss - negs, smalls,
		eis.size());

	double det = 1.0;
	std::sort(evs.begin(), evs.end());
	for (int i = 0; i < evs.size(); i++) {
		printf("%.6f ", evs[i]);
		if (i % 40 == 0 && i > 0)
			printf("\n");
		det *= evs[i];

	}
	printf("\n");
	printf("det: %.10f\n\n", det);
	//cout << eigen_solver.eigenvalues() << endl;
}

void printCSR(const CSR& A) {
	printf("A [ %s ] [ %s ]:\n", A.IsRowMajor ? "CSR" : "CSC", A.isCompressed() ? "Compressed" : "Not Compressed");
	cout << A << endl;
	printf("\n");


	printf("outer size: %d\n", A.outerSize());
	printf("inner size: %d\n", A.innerSize());
	printf("\n");

	printf("outer index, rows: %d + 1\n", A.outerSize());
	for (int i = 0; i < A.outerSize() + 1; i++)
		printf("  %2d  %2d\n", i, A.outerIndexPtr()[i]);

	printf("inner index, columns: %d\n", A.nonZeros());
	for (int i = 0; i < A.nonZeros(); i++)
		printf("  %2d  %2d\n", i, A.innerIndexPtr()[i]);

	printf("values, non-zeros: %d\n", A.nonZeros());
	for (int i = 0; i < A.nonZeros(); i++)
		printf("  %2d  %ef\n", i, A.valuePtr()[i]);

	if (!A.isCompressed()) {
		printf("inner nonzero index:\n");
		for (int j = 0; j < A.outerSize(); j++)
			printf("  %2d  %2d\n", j, A.innerNonZeroPtr()[j]);
	}
	printf("\n\n");
}

void printCSC(const CSC& A) {
	printf("A [ %s ] [ %s ]:\n", A.IsRowMajor ? "CSR" : "CSC", A.isCompressed() ? "Compressed" : "Not Compressed");
	cout << A << endl;
	printf("\n");


	printf("outer size: %d\n", A.outerSize());
	printf("inner size: %d\n", A.innerSize());
	printf("\n");

	printf("outer index, rows: %d + 1\n", A.outerSize());
	for (int i = 0; i < A.outerSize() + 1; i++)
		printf("  %2d  %2d\n", i, A.outerIndexPtr()[i]);

	printf("inner index, columns: %d\n", A.nonZeros());
	for (int i = 0; i < A.nonZeros(); i++)
		printf("  %2d  %2d\n", i, A.innerIndexPtr()[i]);

	printf("values, non-zeros: %d\n", A.nonZeros());
	for (int i = 0; i < A.nonZeros(); i++)
		printf("  %2d  %ef\n", i, A.valuePtr()[i]);

	if (!A.isCompressed()) {
		printf("inner nonzero index:\n");
		for (int j = 0; j < A.outerSize(); j++)
			printf("  %2d  %2d\n", j, A.innerNonZeroPtr()[j]);
	}
	printf("\n\n");
}

void buildEquation2(CSR& A, Vec& b) {
	int M = 4, N = 5;
	CSR A_(M, N);
	Vec b_;
	b_.resize(N);

	A_.insert(1, 1) = 3.0;
	A_.insert(1, 3) = 5.0;
	A_.insert(2, 3) = 6.0;
	A_.insert(3, 1) = 7.0;
	A_.insert(3, 3) = 4.0;

	printCSR(A_);

	for (int i = 0; i < N; i++)
		b_(i) = 1;

	A = A_;
	b = b_;
	//eigenValues(A);
}

void buildEquation(CSR& A, Vec& b) {
	const int div = 5;
	CSR A_(div, div);
	Vec b_;
	b_.resize(div);
	for (int i = 0; i < div; i++) {
		A_.insert(i, i) = i + 1;
		b_(i) = 1;
	}
	A = A_;
	b = b_;
	//eigenValues(A);
}

Vec chol(const CSC& A, const Vec& b) {
	Eigen::SimplicialLLT<CSC> solver;
	printf("CHOL    A: %d x %d, %s,  b: %d\n", A.rows(), A.cols(), A.IsRowMajor ? "CSR" : "CSC", b.size());

	long t0_ = clock();
	solver.analyzePattern(A);
	printf("analyze:  %d ms\n", clock() - t0_);
	solver.factorize(A);
	printf("factorize:  %d ms\n", clock() - t0_);	

	Vec x = solver.solve(b);

	printf("solve -> %d, [ %d ] ms\n", solver.info(), clock() - t0_);
#ifdef PRINT_X
	printf("x:\n");
	printVector(x);
#endif
	printf("\n\n");

	return x;
}

Vec lu(const CSC& A, const Vec& b) {
	Eigen::SparseLU<CSC> solver;
	printf("LU    A: %d x %d, %s,  b: %d\n", A.rows(), A.cols(), A.IsRowMajor ? "CSR" : "CSC", b.size());

	long t0_ = clock();
	solver.analyzePattern(A);
	printf("analyze:  %d ms\n", clock() - t0_);
	solver.factorize(A);
	printf("factorize:  %d ms\n", clock() - t0_);

	Vec x = solver.solve(b);

	printf("solve -> %d, [ %d ] ms\n", solver.info(), clock() - t0_);
#ifdef PRINT_X
	printf("x:\n");
	printVector(x);
#endif
	printf("\n\n");
	
	return x;
}

Vec qr(const CSC& A, const Vec& b) {
	Eigen::SparseQR<CSC, Eigen::AMDOrdering<int>> solver;
	printf("QR    A: %d x %d, %s,  b: %d\n", A.rows(), A.cols(), A.IsRowMajor ? "CSR" : "CSC", b.size());

	long t0_ = clock();
	solver.analyzePattern(A);
	printf("analyze:  %d ms\n", clock() - t0_);
	solver.factorize(A);
	printf("factorize:  %d ms\n", clock() - t0_);

	Vec x = solver.solve(b);	

	printf("solve -> %d, [ %d ] ms\n", solver.info(), clock() - t0_);
#ifdef PRINT_X
	printf("x:\n");
	printVector(x);
#endif
	printf("\n\n");

	return x;
}

Vec lscg(const CSC& A, const Vec& b) {
	Eigen::LeastSquaresConjugateGradient<CSC> solver;
	printf("LSCG    A: %d x %d, %s,  b: %d\n", A.rows(), A.cols(), A.IsRowMajor ? "CSR" : "CSC", b.size());

	solver.setTolerance(CGTOL);
	solver.setMaxIterations(CGMAXITER);
	printf("tol: %.8f, max iter: %d\n", CGTOL, CGMAXITER);

	long t0_ = clock();
	solver.analyzePattern(A);
	printf("analyze:  %d ms\n", clock() - t0_);
	solver.factorize(A);
	printf("factorize:  %d ms\n", clock() - t0_);

	Vec x = solver.solve(b);

	printf("solve -> %d, iter: %d, err: %.8f,  [ %d ] ms\n", solver.info(), solver.iterations(), solver.error(), clock() - t0_);
#ifdef PRINT_X
	printf("x:\n");
	printVector(x);
#endif
	printf("\n\n");

	return x;
}

Vec cg(const CSC& A, const Vec& b, const Vec* x0_ = nullptr) {
	//Eigen::ConjugateGradient<CSR, Eigen::Lower|Eigen::Upper> solver;
	Eigen::ConjugateGradient<CSC> solver;
	printf("CG    A: %d x %d, %s, b: %d\n", A.rows(), A.cols(), A.IsRowMajor ? "CSR" : "CSC", b.size());

	solver.setTolerance(CGTOL);
	solver.setMaxIterations(CGMAXITER);
	printf("tol: %.8f, max iter: %d\n", CGTOL, CGMAXITER);

	long t0_ = clock();
	solver.analyzePattern(A);
	printf("analyze:  %d ms\n", clock() - t0_);
	solver.factorize(A);
	printf("factorize:  %d ms\n", clock() - t0_);

	Vec x0;
#if 1
	if (nullptr == x0_) {
		x0.resize(b.size());
		x0.setZero();
	}
	else x0 = *x0_;
#endif
	
	Vec x = solver.solveWithGuess(b, x0);

	printf("solve -> %d, iter: %d, tol: %ef, err: %ef,  [ %d ] ms\n", solver.info(), solver.iterations(), solver.tolerance(), solver.error(), clock() - t0_);
#ifdef PRINT_X
	printf("x:\n");
	printVector(x);	
#endif
	printf("\n\n");

	return x;
}

double norm(const Vec& v) {
	double sum = 0.0;
	for (int i = 0; i < v.size(); i++)
		sum += v(i)* v(i);
	return sqrt(sum);
}

double cosd(const Vec& v1, const Vec& v2) {
	double norm1 = norm(v1);
	double norm2 = norm(v2);

	double dot = 0;
	for (int i = 0; i < v1.size(); i++)
		dot += v1(i) * v2(i);

	return sqrt(dot) / (norm1 * norm2);
}


/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int* I, int* J, float* val, int N, int nz)
{
	I[0] = 0, J[0] = 0, J[1] = 1;
	val[0] = (float)rand() / RAND_MAX + 10.0f;
	val[1] = (float)rand() / RAND_MAX;
	int start;

	for (int i = 1; i < N; i++)
	{
		if (i > 1)
		{
			I[i] = I[i - 1] + 3;
		}
		else
		{
			I[1] = 2;
		}

		start = (i - 1) * 3 + 2;
		J[start] = i - 1;
		J[start + 1] = i;

		if (i < N - 1)
		{
			J[start + 2] = i + 1;
		}

		val[start] = val[start - 1];
		val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

		if (i < N - 1)
		{
			val[start + 2] = (float)rand() / RAND_MAX;
		}
	}

	I[N] = nz;
}

#define PRINT_CUCG_IJ


void buildEquation(int N, int& nz, int*& I, int*& J, float*& val, float*& b) {	
	nz = (N - 2) * 3 + 4;

	I = (int*)malloc(sizeof(int) * (N + 1));
	J = (int*)malloc(sizeof(int) * nz);
	val = (float*)malloc(sizeof(float) * nz);

	genTridiag(I, J, val, N, nz);

#ifdef PRINT_CUCG_IJ
	printf("I, %d + 1:\n", N);
	for (int i = 0; i < N + 1; i++)
		printf("%2d %4d\n", i, I[i]);
	printf("\n");

	printf("J & value, %d:\n", nz);
	for (int j = 0; j < nz; j++)
		printf("%2d [ %2d ]  %ef\n", j, J[j], val[j]);

	printf("\n");
#endif

	b = (float*)malloc(sizeof(float) * N);
	for (int i = 0; i < N; i++)
		b[i] = i + 1;
}

void printMatrix(int N, int nz, int* I, int* J, float* val) {
	printf("CSR:\n");
	int p = 0;
	for (int r = 0; r < N; r++) {
		int eprs = I[r + 1] - I[r];
		if (eprs > 0) {
			for (int i = 0; i < N; i++) {
				if (J[p] == i) {
					printf("  %4.1f ", val[p]);
					p++;
				}
				else printf("  %4.1f ", 0);
			}
			printf("\n");
		}
		else {
			for (int i = 0; i < N; i++)
				printf("  %4.1f ", 0);
			printf("\n");
		}
	}
	printf("\n");
}

void dumpMatrix(int N, int nz, int* I, int* J, float* val) {
	char sz[256];
	sprintf(sz, "tmp/A-%d.txt", N);
	FILE* fp = fopen(sz, "w");
	fprintf(fp, "%d\n", N);
	fprintf(fp, "%d\n", N);
	fprintf(fp, "%d\n", nz);
	int p = 0;
	for (int r = 0; r < N; r++) {
		int eprs = I[r + 1] - I[r];
		if (eprs > 0) {
			for (int i = 0; i < N; i++) {
				if (J[p] == i) {
					fprintf(fp, "%d %d  %ef\n", r, i, val[p]);
					p++;
				}		
			}		
		}
	}
	fclose(fp);
}

void dumpVector(float* b, int N) {
	char sz[256];
	sprintf(sz, "tmp/b-%d.txt", N);
	FILE* fp = fopen(sz, "w");
	fprintf(fp, "%d\n", N);
	for (int i = 0; i < N; i++)
		fprintf(fp, "%ef\n", b[i]);
	fclose(fp);
}

Vec toVec(float* b, int N) {
	Vec v;
	v.resize(N);
	for (int i = 0; i < N; i++)
		v[i] = b[i];
	return v;
}

CSR toCSR(int N, int nz, int* I, int* J, float* val) {
	CSR A(N, N);
	int p = 0;
	for (int r = 0; r < N; r++) {
		if (I[r + 1] - I[r] > 0) {
			for (int c = 0; c < N; c++) {
				if (J[p] == c) {
					A.insert(r, c) = val[p];
					p++;
				}
			}
		}
	}
	return A;
}

int cucgTest()
{
	/* Generate a random tridiagonal symmetric matrix in CSR format */
	int N = 15, nz, * I, * J;
	float* val, * b;
	buildEquation(N, nz, I, J, val, b);
	auto A_ = toCSR(N, nz, I, J, val);
	auto b_ = toVec(b, N);

#ifdef PRINT_CUCG_IJ
	printMatrix(N, nz, I, J, val);
	printVector(b, N);

	CSC A__ = A_;

	printCSR(A__);
	printCSR(A_);
#endif
	dumpMatrix(N, nz, I, J, val);
	dumpVector(b, N);


#if 1
	chol(A_, b_);
	lu(A_, b_);
	cg(A_, b_);
#endif
	cucg(N, nz, I, J, val, b);

	free(I);
	free(J);
	free(val);
	free(b);

	system("PAUSE");
	return 0;
}

#if 0
void openmpTest() {
	int nProcessors = omp_get_max_threads();

	std::cout << nProcessors << std::endl;

	omp_set_num_threads(nProcessors);

	std::cout << omp_get_num_threads() << std::endl;

#pragma omp parallel for 
	for (int i = 0; i < 10; i++) {		
		printf("%d / %d\n", omp_get_thread_num(), omp_get_num_threads());
	}
}
#endif
void mklTest() {
	printf("MKL test, mkl: %s\n", mkl > 0 ? "open" : "close");

	auto t0_ = clock();
	auto m1 = MatrixXd::Random(20000001, 30000000);
	auto m2 = MatrixXd::Random(30000000, 30000003);
	auto m3 = m1 * m2;

	printf("%d %d x %d %d = %d %d\n", m1.rows(), m1.cols(), m2.rows(), m2.cols(), m3.rows(), m3.cols());
	auto t0 = clock() - t0_;

	printf("%d ms\n", t0);
}

void solverTest() {
#ifdef redirect
	FILE* fp = freopen("log.txt", "w", stdout);
#endif

#if 1
	#if 0
		const char* Afile = "tmp/A-15.txt";
		const char* bfile = "tmp/b-15.txt";
	#elif 0
		const char* Afile = "tmp/q2340.txt";
		const char* bfile = "tmp/h2340.txt";
	#else
		const char* Afile = "tmp/q606.txt";
		const char* bfile = "tmp/h606.txt";
	#endif
	CSC A;
	Vec b;
	readEquation(A, b, Afile, bfile);
	//eigenValues(A);
	#if 1
		Vec x0 = lscg(A, b);
		Vec x3 = chol(A, b);
		Vec x1 = cg(A, b);
		Vec x2 = lu(A, b);	
		Vec x4 = qr(A, b);
	#endif

	CSR A_ = A;
	cucg(A_.rows(), A_.nonZeros(), A_.outerIndexPtr(), A_.innerIndexPtr(), A_.valuePtr(), b.data());
	//cucggraph(A_.rows(), A_.nonZeros(), A_.outerIndexPtr(), A_.innerIndexPtr(), A_.valuePtr(), b.data());
	cupcg(A_.rows(), A_.nonZeros(), A_.outerIndexPtr(), A_.innerIndexPtr(), A_.valuePtr(), b.data());

#if 0
	printf("cos 1 & 2: %f\n", cosd(x1, x2));
	printf("cos 1 & 3: %f\n", cosd(x1, x3));
	printf("cos 1 & 4: %f\n", cosd(x1, x4));
	printf("cos 2 & 3: %f\n", cosd(x2, x3));
	printf("cos 2 & 4: %f\n", cosd(x2, x4));
	printf("cos 3 & 4: %f\n", cosd(x3, x4));
#endif


#else
	cucgTest();
#endif	


#if 0;
	CSR A2;
	Vec b2;
	buildEquation(A2, b2);

	//cg2(A2, b2);
	cg(A2, b2);
	chol(A2, b2);

	CSR A3;
	Vec b3;
	buildEquation2(A3, b3);
#endif

	//cucg();	

#ifdef redirect
	fflush(fp);
	fclose(fp);
	freopen("CON", "w", stdout);
#endif
}


int main(int argc, char* argv[])
{
#if 0
	CSR A;
	Vec b;
	buildEquation2(A, b);
	CSC A_ = A;
	printCSR(A);
	printCSC(A_);

#else	
	solverTest();
#endif
    return 0;
}

