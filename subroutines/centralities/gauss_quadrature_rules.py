import numpy as np
import numpy.linalg as la


	#####################################################################################################
	#													#
	# DESCRIPTION:	All below methods compute lower and/or upper bounds on quantities of the form	#
	#		u^T f(A) u for vectors u \in R^n, symmetric matrices A \in R^{n \times n}, and	#
	#		matrix functions f by means of a relation between Gauss quadrature, orthogonal	#
	#		polynomials, and the symmetric Lanczos method discussed by Golub and Meurant		#
	#		[1,2,3,4].										#
	#													#
	# FILES												#
	# CREATED:	None											#
	#													#
	# REFERENCES:	[1] G. H. Golub and J. H. Welsch, Calculation of Gauss quadrature rules,		#
	#		Mathematics of Computation, 23 (1969), pp. 221–230,					#
	#		https://doi.org/10.1090/S0025-5718-69-99647-1.					#
	#		[2] G. H. Golub and G. Meurant, Matrices, moments and quadrature, Pitman Research	#	
	#		Notes in Mathematics Series, 303 (1994), pp. 105–156.				#
	#		[3] G. H. Golub and G. Meurant, Matrices, moments and quadrature II; How to compute	#
	#		the norm of the error in iterative methods, BIT Numerical Mathematics, 37 (1997),	#
	#		pp. 687–705, https://doi.org/10.1007/BF02510247.					#
	#		[4] G. H. Golub and G. Meurant, Matrices, Moments and Quadrature with Applications,	#
	#		Princeton University Press, USA, 2009, https://doi.org/10.1515/9781400833887.	#
	#													#
	# 2021, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)						#
	#													#
	#####################################################################################################


def gauss_subgraph(T, beta_subgraph):

	#####################################################################################################
	#													#
	# DESCRIPTION:	lower Gauss quadrature bound on subgraph centrality, i.e., u^T f(A) u for		#
	#		u \in R^n, A \in R^{n \times n} symmetric, and f(A) = exp(beta_subgraph*A) the	#
	#		matrix exponential.									#
	#													#
	# INPUT: 	T (required): matrix in R^{k \times k} of tridiagonal form obtained by k steps of	#
	#			the Lanczos method applied to the matrix A and the vector u.			#
	#		beta_subgraph (required): scalar parameter from the matrix function			#
	#			f(A) = exp(beta_subgraph*A).							#
	#													#
	# OUTPUT:	int_val (float): (scalar) lower bound on u^T f(A) u.					#
	#													#
	#####################################################################################################

	k, k = T.shape
	if k==1:
		int_val=1
	else:
		lamb, phi = la.eig(T[0:k-1, 0:k-1])
		e = np.zeros([k-1,1])
		e[0] = 1
		int_val = (np.dot(e.T, phi).dot(np.diag(np.exp(beta_subgraph*lamb))).dot(phi.T).dot(e)).item()

	return int_val


def gauss_radau_subgraph(T, beta_subgraph, lambda_bound):

	#####################################################################################################
	#													#
	# DESCRIPTION:	lower or upper Gauss--Radau quadrature bound on subgraph centrality, i.e., 		#
	#		u^T f(A) u for u \in R^n, A \in R^{n \times n} symmetric, and			#
	#		f(A) = exp(beta_subgraph*A) the matrix exponential. 'lambda_bound' is prescribed to	#
	#		the matrix T \in R^{k \times k} as an eigenvalue and can be chosen to be the		#
	#		smallest or the largest eigenvalue of A. In case of the smallest eigenvalue we	#
	#		obtain a lower bound and in case of the largest eigenvalue we obtain an upper bound	#
	#		on u^T f(A) u.										#
	#													#
	# INPUT: 	T (required): matrix in R^{k \times k} of tridiagonal form obtained by k steps of	#
	#			the Lanczos method applied to the matrix A and the vector u.			#
	#		beta_subgraph (required): scalar parameter from the matrix function			#
	#			f(A) = exp(beta_subgraph*A).							#
	#		lambda_bound (required): smallest or largest eigenvalue of A. Smallest eigenvalue	#
	#			leads to a lower bound and largest eigenvalue to an upper bound on		#
	#			u^T f(A) u.									#
	#													#
	# OUTPUT:	int_val (float): (scalar) lower or upper bound on u^T f(A) u.			#
	#													#
	#####################################################################################################
	
	k, k = T.shape
	if k==1:
		int_val=1
	else:
		rhs = np.zeros([k-1,1])
		rhs[k-2] = T[k-2, k-1]**2
		e = np.zeros([k,1])
		e[0] = 1

		# prescribe the eigenvalue lambda_bound to T
		delta = la.solve(T[0:k-1, 0:k-1]-lambda_bound*np.eye(k-1), rhs)
		T[k-1,k-1] = lambda_bound + delta[k-2]

		lamb, phi = la.eig(T)
		int_val = (np.dot(e.T, phi).dot(np.diag(np.exp(beta_subgraph*lamb))).dot(phi.T).dot(e)).item()

	return int_val

def gauss_lobatto_subgraph(T, beta_subgraph, lambda_min, lambda_max):

	#####################################################################################################
	#													#
	# DESCRIPTION:	upper Gauss--Lobatto quadrature bound on subgraph centrality, i.e., u^T f(A) u for	#
	#		u \in R^n, A \in R^{n \times n} symmetric, and f(A) = exp(beta_subgraph*A) the	#
	#		matrix exponential. 'lambda_min' and 'lambda_max' are prescribed to the matrix	#
	#		T \in R^{k \times k} as eigenvalues.							#
	#													#
	# INPUT: 	T (required): matrix in R^{k \times k} of tridiagonal form obtained by k steps of	#
	#			the Lanczos method applied to the matrix A and the vector u.			#
	#		beta_subgraph (required): scalar parameter from the matrix function			#
	#			f(A) = exp(beta_subgraph*A).							#
	#		lambda_min (required): smallest eigenvalue of A.					#
	#		lambda_max (required): largest eigenvalue of A.					#
	#													#
	# OUTPUT:	int_val (float): (scalar) lower or upper bound on u^T f(A) u.			#
	#													#
	#####################################################################################################

	k, k = T.shape
	if k==1:
		int_val=1
	else:
		e_k = np.zeros([k-1,1])
		e_k[k-2] = 1
		e = np.zeros([k,1])
		e[0] = 1

		# prescribe both eigenvalues to T
		delta = la.solve(T[0:k-1, 0:k-1]-lambda_min*np.eye(k-1), e_k)
		mu = la.solve(T[0:k-1, 0:k-1]-lambda_max*np.eye(k-1), e_k)
		T_entries = la.solve([[1, (-delta[k-2]).item()], [1, (-mu[k-2]).item()]], [[lambda_min], [lambda_max]])
		T[k-1, k-1] = T_entries[0]

		# catch error
		if T_entries[1] <= 0:
			int_val = 1
			print('Warning, prevented taking the root of %f in gauss_lobatto_subgraph. Set centrality value to 1.' % (T_entries[1]).item())
		else:
			T[k-2, k-1] = np.sqrt(T_entries[1])
			T[k-1, k-2] = np.sqrt(T_entries[1])

			lamb, phi = la.eig(T)
			int_val = (np.dot(e.T, phi).dot(np.diag(np.exp(beta_subgraph*lamb))).dot(phi.T).dot(e)).item()

	return int_val

def gauss_resolvent(T, alpha_resolvent):

	#####################################################################################################
	#													#
	# DESCRIPTION:	lower Gauss quadrature bound on resolvent-based subgraph centrality, i.e.,		#
	#		u^T f(A) u for u \in R^n, A \in R^{n \times n} symmetric, and			#
	#		f(A) = (I - alpha_resolvent*A)^{-1} the matrix resolvent function.			#
	#													#
	# INPUT: 	T (required): matrix in R^{k \times k} of tridiagonal form obtained by k steps of	#
	#			the Lanczos method applied to the matrix A and the vector u.			#
	#		alpha_resolvent (required): scalar parameter from the matrix function		#
	#			f(A) = (I - alpha_resolvent*A)^{-1}.						#
	#													#
	# OUTPUT:	int_val (float): (scalar) lower bound on u^T f(A) u.					#
	#													#
	#####################################################################################################

	k, k = T.shape
	if k==1:
		int_val=1
	else:
		lamb, phi = la.eig(T[0:k-1, 0:k-1])
		e = np.zeros([k-1,1])
		e[0] = 1
		int_val = (np.dot(e.T, phi).dot(la.solve((np.eye(k-1) - alpha_resolvent*np.diag(lamb)), np.eye(k-1))).dot(phi.T).dot(e)).item()

	return int_val

def gauss_radau_resolvent(T, alpha_resolvent, lambda_bound):

	#####################################################################################################
	#													#
	# DESCRIPTION:	lower or upper Gauss--Radau quadrature bound on resolvent-based subgraph		#
	#		centrality, i.e., u^T f(A) u for u \in R^n, A \in R^{n \times n} symmetric, and	#
	#		f(A) = (I - alpha_resolvent*A)^{-1} the matrix resolvent function. 'lambda_bound'	#
	#		is prescribed to the matrix T \in R^{k \times k} as an eigenvalue and can be	chosen	#
	#		to be the smallest or the largest eigenvalue of A. In case of the smallest		#
	#		eigenvalue we obtain a lower bound and in case of the largest eigenvalue we obtain	#
	#		an upper bound on u^T f(A) u.								#
	#													#
	# INPUT: 	T (required): matrix in R^{k \times k} of tridiagonal form obtained by k steps of	#
	#			the Lanczos method applied to the matrix A and the vector u.			#
	#		alpha_resolvent (required): scalar parameter from the matrix function		#
	#			f(A) = (I - alpha_resolvent*A)^{-1}.						#
	#		lambda_bound (required): smallest or largest eigenvalue of A. Smallest eigenvalue	#
	#			leads to a lower bound and largest eigenvalue to an upper bound on		#
	#			u^T f(A) u.									#
	#													#
	# OUTPUT:	int_val (float): (scalar) lower or upper bound on u^T f(A) u.			#
	#													#
	#####################################################################################################
	
	k, k = T.shape
	if k==1:
		int_val=1
	else:
		rhs = np.zeros([k-1,1])
		rhs[k-2] = T[k-2, k-1]**2
		e = np.zeros([k,1])
		e[0] = 1

		 # prescribe the eigenvalue lambda_bound to T
		delta = la.solve(T[0:k-1, 0:k-1]-lambda_bound*np.eye(k-1), rhs)
		T[k-1,k-1] = lambda_bound + delta[k-2]

		lamb, phi = la.eig(T)
		int_val = (np.dot(e.T, phi).dot(la.solve((np.eye(k) - alpha_resolvent*np.diag(lamb)), np.eye(k))).dot(phi.T).dot(e)).item()

	return int_val

def gauss_lobatto_resolvent(T, alpha_resolvent, lambda_min, lambda_max):

	#####################################################################################################
	#													#
	# DESCRIPTION:	upper Gauss--Lobatto quadrature bound on resolvent-based subgraph centrality, i.e.,	#
	#		u^T f(A) u for u \in R^n, A \in R^{n \times n} symmetric, and			#
	#		f(A) = (I - alpha_resolvent*A)^{-1} the matrix resolvent function. 'lambda_min' and	#
	#		'lambda_max' are prescribed to the matrix T \in R^{k \times k} as eigenvalues.	#
	#													#
	# INPUT: 	T (required): matrix in R^{k \times k} of tridiagonal form obtained by k steps of	#
	#			the Lanczos method applied to the matrix A and the vector u.			#
	#		beta_subgraph (required): scalar parameter from the matrix function			#
	#			f(A) = (I - alpha_resolvent*A)^{-1}.						#
	#		lambda_min (required): smallest eigenvalue of A.					#
	#		lambda_max (required): largest eigenvalue of A.					#
	#													#
	# OUTPUT:	int_val (float): (scalar) lower or upper bound on u^T f(A) u.			#
	#													#
	#####################################################################################################

	k, k = T.shape
	if k==1:
		int_val=1
	else:
		e_k = np.zeros([k-1,1])
		e_k[k-2] = 1
		e = np.zeros([k,1])
		e[0] = 1

		# prescribe both eigenvalues to T
		delta = la.solve(T[0:k-1, 0:k-1]-lambda_min*np.eye(k-1), e_k)
		mu = la.solve(T[0:k-1, 0:k-1]-lambda_max*np.eye(k-1), e_k)
		T_entries = la.solve([[1, (-delta[k-2]).item()], [1, (-mu[k-2]).item()]], [[lambda_min], [lambda_max]])
		T[k-1, k-1] = T_entries[0]

		# catch error
		if T_entries[1] <= 0:
			int_val = 1
			print('Warning, prevented taking the root of %f in gauss_lobatto_subgraph. Set centrality value to 1.' % (T_entries[1]).item())
		else:
			T[k-2, k-1] = np.sqrt(T_entries[1])
			T[k-1, k-2] = np.sqrt(T_entries[1])

			lamb, phi = la.eig(T)
			int_val = (np.dot(e.T, phi).dot(la.solve((np.eye(k) - alpha_resolvent*np.diag(lamb)), np.eye(k))).dot(phi.T).dot(e)).item()

	return int_val

