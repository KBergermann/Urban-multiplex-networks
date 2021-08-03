import numpy as np
import numpy.linalg as la


def expAb_sym(U, T, b, beta_subgraph):

	#####################################################################################################
	#													#
	# DESCRIPTION:	approximates the quantity f(A)b [1] for a symmetric matrix A \in R^{n \times n}, a	#
	#		vector b \in R^n, and f(A) = exp(beta_subgraph*A) the matrix exponential. The 	#
	#		function requires a basis U \in R^{n x k} of the Krylov subspace of A to the vector	#
	#		b and the corresponding tridiagonal matrix T \in R^{k \times k} so that we obtain	#
	#		a matrix factorization A \approx U T U^T. U and T can be computed by the Lanczos	#
	#		method [2,3], which is implemented in the function 'lanczos' in the same directory.	#
	#													#
	# INPUT: 	U (required): matrix in R^{n x k}, which contains the first k basis vectors of the	#
	#			Krylov subspace of the matrix A to the vector b.				#
	#		T (required): matrix in R^{k \times k} of tridiagonal form for which holds: 		#
	#			T \approx U^T A U.								#
	#		b (required): vector in R^n from the quantity f(A)b, which we seek to approximate.	#
	#		beta_subgraph (required): scalar parameter in the matrix exponential			#
	#			f(A) = exp(beta_subgraph*A).							#
	#													#
	# OUTPUT: 	fAb (numpy.ndarray): vector in R^n, which corresponds to the approximation of	#
	#			f(A)b.										#
	#													#
	# FILES												#
	# CREATED:	None											#
	#													#
	# REFERENCE:	[1] N. J. Higham, Functions of Matrices: Theory and Computation, SIAM, USA, 2008,	#
	#		https://doi.org/10.1137/1.9780898717778.						#
	#		[2] C. Lanczos, An iteration method for the solution of the eigenvalue problem of	#
	#		linear differential and integral operators, United States Governm. Press Office	#
	#		Los Angeles, CA, USA, 1950, https://doi.org/10.6028/jres.045.026.			#
	#		[3] G. H. Golub and C. F. Van Loan, Matrix Computations, vol. 3, JHU press, USA,	#
	#		2013.											#
	#													#
	#####################################################################################################
	
	lamb, phi = la.eig(T)
	UTb = np.dot(U.T, b)
	fAb = U.dot(phi).dot(np.diag(np.exp(beta_subgraph*lamb))).dot(phi.T).dot(UTb)
	return fAb

def resolventAb_sym(U, T, b, alpha_resolvent):

	#####################################################################################################
	#													#
	# DESCRIPTION:	approximates the quantity f(A)b [1] for a symmetric matrix A \in R^{n \times n}, a	#
	#		vector b \in R^n, and f(A) = (I - alpha_resolvent*A)^{-1} the matrix resolvent	#
	#		function, where I \in R^{n \times n} denotes the identity matrix and ^{-1} the	#
	#		matrix inverse. The function requires a basis U \in R^{n x k} of the Krylov		#
	#		subspace of A to the vector b and the corresponding tridiagonal matrix		#
	#		T \in R^{k \times k} so that we obtain a matrix factorization A \approx U T U^T.	#
	#		U and T can be computed by the Lanczos method [2,3], which is implemented in the	#
	#		function 'lanczos' in the same directory.						#
	#													#
	# INPUT: 	U (required): matrix in R^{n x k}, which contains the first k basis vectors of the	#
	#			Krylov subspace of the matrix A to the vector b.				#
	#		T (required): matrix in R^{k \times k} of tridiagonal form for which holds: 		#
	#			T \approx U^T A U.								#
	#		b (required): vector in R^n from the quantity f(A)b, which we seek to approximate.	#
	#		alpha_resolvent (required): scalar parameter in the matrix resolvent function	#
	#			f(A) = (I - alpha_resolvent*A)^{-1}.						#
	#													#
	# OUTPUT: 	fAb (numpy.ndarray): vector in R^n, which corresponds to the approximation of	#
	#			f(A)b.										#
	#													#
	# REFERENCE:	[1] N. J. Higham, Functions of matrices: Theory and computation, SIAM, USA, 2008,	#
	#		https://doi.org/10.1137/1.9780898717778.						#
	#		[2] C. Lanczos, An iteration method for the solution of the eigenvalue problem of	#
	#		linear differential and integral operators, United States Governm. Press Office	#
	#		Los Angeles, CA, USA, 1950, https://doi.org/10.6028/jres.045.026.			#
	#		[3] G. H. Golub and C. F. Van Loan, Matrix computations, vol. 3, JHU press, USA,	#
	#		2013.											#
	#													#
	# 2021, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)						#
	#													#
	#####################################################################################################

	k, k = T.shape
	lamb, phi = la.eig(T)
	UTb = np.dot(U.T, b)
	fAb = U.dot(phi).dot(la.solve((np.eye(k) - alpha_resolvent*np.diag(lamb)), np.eye(k))).dot(phi.T).dot(UTb)
	return fAb
	
