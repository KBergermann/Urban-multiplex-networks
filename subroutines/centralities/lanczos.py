import scipy.sparse as spsp
import numpy as np
import numpy.linalg as la


def lanczos(A, u, maxit):

	#####################################################################################################
	#													#
	# DESCRIPTION:	computes the first k <= 'maxit' basis vectors of the Krylov subspace of a symmetric	#
	#		matrix A \in R^{n \times n} to a vector u \in R^n using the Lanczos method [1,2].	#
	#		The method results in a matrix decomposition A \approx U T U^T, where		#
	#		U \in R^{n \times k} contains the first k Krylov subspace basis vectors and		#
	#		T \in R^{k \times k} has tridiagonal form. The method terminates in less than	#
	#		'maxit' iterations, if the decomposition A = U T U^T is 'exact' for k < 'maxit'.	#
	#													#
	# INPUT: 	A (required): symmetric matrix in R^{n \times n}.					#
	#		u (required): vector in R^n.								#
	#		maxit (required): integer denoting the maximal number of Lanczos iterations.		#
	#													#
	# OUTPUT:	U (numpy.ndarray): matrix in R^{n x k}, which contains the first k basis vectors of	#
	#			the Krylov subspace of the matrix A to the vector b.				#
	#		T (numpy.ndarray): matrix in R^{k \times k} of tridiagonal form for which holds:	#
	#			T \approx U^T A U.								#
	#													#
	# FILES												#
	# CREATED:	None											#
	#													#
	# REFERENCES:	[1] C. Lanczos, An iteration method for the solution of the eigenvalue problem of	#
	#		linear differential and integral operators, United States Governm. Press Office	#
	#		Los Angeles, CA, USA, 1950, https://doi.org/10.6028/jres.045.026.			#
	#		[2] G. H. Golub and C. F. Van Loan, Matrix Computations, vol. 3, JHU press, USA,	#
	#		2013.											#
	#													#
	# 2021, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)						#
	#													#
	#####################################################################################################

	nL, nL = A.shape
	u = u/la.norm(u)
	U = u
	alpha = []
	beta = []
	T = np.zeros((1,1))
	for j in range(maxit):
		if j==0:
			U = np.hstack([U, A.dot(U[:,[j]])])
		else:
			U = np.hstack([U, np.subtract(A.dot(U[:,[j]]), beta[j-1]*U[:,[j-1]])])
		alpha.append(np.asscalar(np.dot(U[:,[j+1]].T, U[:,[j]])))
		U[:,[j+1]] = np.subtract(U[:,[j+1]], alpha[j]*U[:,[j]])
		beta.append(la.norm(U[:,[j+1]]))

		if abs(beta[j]) < 1e-14:
			#print('Warning: Symmetric Lanczos broke down with beta=0 and\nT=', T)
			break

		U[:,[j+1]] = U[:,[j+1]]/beta[j]

		T = np.hstack([T, np.zeros((j+1,1))])
		T = np.vstack([T, np.zeros((1,j+2))])
		T[j,j] = alpha[j]
		T[j, j+1] = beta[j]
		T[j+1, j] = beta[j]

	return U, T
	
