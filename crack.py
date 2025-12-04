"""Imports"""
import numpy as np
import math
import galois
import itertools as it




"""Key Generation Function - used for testing purposes"""
"""
    - n: the length of the private key, s
    - m: the number of LWE equations, ie. # of rows in A and e
    - q: the modulo group we work in, Zq
"""
def key_gen(n, m = 0, q = 0):
    # Initialise q ~ n^2 if q not given
    if not q:
        q = round(n ** 2)

    # Initialise m ~ n.logq if m not given
    if not m:
        m = round(n * math.log(q))

    # Randomly generate A from Zq:
    A = np.random.randint(0, q, size = (m, n))

    # Randomly generate s from Zq:
    s = np.random.randint(0, q, size = n)

    # Randomly generate e from Gaussian distribution
    e = np.round(np.random.normal(0, 0.5, m)).astype(int) 

    # Calculate b = A.s + e
    b = np.mod((np.matmul(A, s) + e), q)

    # Return (public_key, private_key) pair
    return ((A, b), s)




"""LWE encryption function"""
"""
    - plaintext: bit string to be encrypted
    - public_key: of the form (A, b) 
    - q: the modulo group we work in, Zq
"""
def encrypt(plaintext, public_key, q):
    # Initialise empty ciphertext as object type numpy array
    ciphertext = np.empty(len(plaintext), dtype=object)

    # Encrypt for each bit in the plaintext
    for i in range(len(plaintext)):

        # Randomly generate r, ensuring it is never all zero (since we need to use at least 1 row of A for the encryption)
        r = np.random.randint(0, 2, size = public_key[0].shape[0])
        while not np.any(r):
            r = np.random.randint(0, 2, size = public_key[0].shape[0])

        # a'^T = r^T . A
        a1_t = np.mod(np.matmul(r, public_key[0]), q)

        # b' = r^T . b + pt . q/2
        b1 = np.mod((np.matmul(r, public_key[1])) + (plaintext[i] * math.floor(q / 2)), q)

        # Append the ciphertext bit pair (a', b')
        ciphertext[i] =  (a1_t, b1)
        
    # Return the ciphertext object array
    return ciphertext





"""LWE decryption function"""
"""
    - ciphertext: array of a' along with bit strings to be decrypted 
    - private_key: s, used to decrypt ciphertext
    - q: the modulo group we work in, Zq
"""
def decrypt(ciphertext, private_key, q):

    # Initialise empty plaintext as int type numpy array
    plaintext = np.empty(len(ciphertext), dtype=int)

    # Decrypt for each bit in the ciphertext
    for i in range(len(ciphertext)):

        # Extract (a', b') for current bit
        a1, b1 = ciphertext[i]

        # v = a'^T . s
        v = np.mod(np.matmul(a1, private_key), q)

        # m' = b' - v
        m1 = np.mod(b1 - v, q)

        # Bring m' between 0 and q/2, rather than dealing separately with m' > q/2
        if m1 < q - m1: 
            # Take m', decide if it is closer to 0 or q/2
            if m1 <= abs(m1 - (q / 2)):
                # 0
                plaintext[i] = 0
            else: 
                # 1
                plaintext[i] = 1
        
        else:
            # Take q - m', decide if it is closer to 0 or q/2
            if q - m1 <= abs(m1 - (q / 2)):
                # 0
                plaintext[i] = 0
            else: 
                # 1
                plaintext[i] = 1
    
    # Return the plaintext array
    return plaintext






    



"""Crack 1 - Learning Without Errors"""
"""
    - ciphertext: array of a' along with bit strings to be decrypted 
    - public_key: of the form (A, b) 
    - q: the modulo group we work in, Zq
"""
def crack1(ciphertext, public_key, q):
    # Create Galois Field object to work in modulo q
    GF = galois.GF(q)

    # Extract A, b from the public key and initialise them as Zq field
    A, b = public_key
    A = GF(A)
    b = GF(b)

    # Since there are no errors, we have that:
    # b = A.s + e   ->   b = A.s   ->   A_left . b = s
    # ...where A_left is the left inverse of A
    # Certain conditions must be met for this to work:
    #   1. Rank of A must equal number of columns in A (ie, rank(A) = n)
    #   2. Number of rows must be greater than number of columns (ie, m > n)
    #       - This will almost certainly hold since m is usually chosen to be much larger than n for the decryption to work with errors
    #       - The randomness of A at this size also typically ensures linearly independent rows/columns, making this approach very likely to work

    # Compute A^T, A transposed
    At = np.transpose(A)

    # Compute A^T . A
    At_A = np.matmul(At, A)

    # Compute the inverse of A^T . A
    # Note that this can only be done if the det(A^T . A) is not 0, and if every element in it has multiplicative inverse modulo q 
    #   - This will be the case if every element in A^T . A is coprime to q, (for all x in A^T . A: gcd(x, q) = 1)
    #   - This will always be the case if q is prime, since trivially no number less than q (except 1 itself) can divide q if q is prime
    #   - det(A^T . A) non zero since rank = n
    At_A_inv = np.linalg.inv(At_A)

    # Compute A_left as (A^T . A)^-1 . A^T
    A_left = np.matmul(At_A_inv, At)

    # Calculate s as above
    s = np.matmul(A_left, b)

    # Convert s back from the Galois field object into a numpy array
    s = np.array(s)

    # Return the result of decrypting the ciphertext given with the private key s we have just computed
    return decrypt(ciphertext, s, q)





"""Crack 2 - Learning With a Few Errors"""
"""
    - ciphertext: array of a' along with bit strings to be decrypted 
    - public_key: of the form (A, b) 
    - q: the modulo group we work in, Zq
"""
def crack2(ciphertext, public_key, q):
    # Start with b = A.s + e
    # We are going to brute force e, on the basis that e is typically mostly zero so it shouldn't take too long
    #    do b - e = A.s
    #    A_left.(b - e) = s, 
    # This is the private key assuming we have the right e

    # Now going back to the lattice basis, we expect the shortest vector to be that of the error
    # ESPECIALLY, since the error vector is even smaller now that there are fewer errors than the standard LWE encryption (seen in crack 3)
    #   |A b| |s |    =   | A.s - b |
    #   |0 1| |-1|        |   -1    |

    # This should give the shortest vector which is (-e -1)
    # Check that -1 * e that we brute forced is equal to this -e
    # If this is the case, we end the brute force search and extract the s with this e
    
    # Essentially we are checking if A.(A_left.(b - e)) - b = -e 

    #to iterate through e's, we start with all zeroes, then all perms of 1 1, then all perms of 1 -1, then all perms of 2 errors (1 or -1 for each as option)

    #use mCx, x starts as 0 errors, 1 error, 2 errors (combinations not perms to find the slots) (within selected slots, 2 optoon for each slot so 2**num of slots)

    # Create Galois Field object to work in modulo q
    GF = galois.GF(q)

    # Extract A, b from the public key and initialise them as Zq field
    A, b = public_key
    A = GF(A)
    b = GF(b)

    # Compute A_left (constant for each guess of e)
    At = np.transpose(A)
    At_A = np.matmul(At, A)
    At_A_inv = np.linalg.inv(At_A)
    A_left = np.matmul(At_A_inv, At)

    # Reshape b
    b = b.reshape((len(b), 1))

    # Contain index references for this size of e (m - calculated from rows in A from public key)
    indeces = list(range(0, len(A)))

    # Initialise brute force search parameters
    #   found: True only if we have found a matching e
    #   finished: True only if we have exhausted all options for e 
    #   r: The number of errors we are permitting in e (0, 1, ...)
    #       - It is important we work ascendingly from r = 0 since there won't be very many errors
    found = False
    finished = False
    r = 0

    # Continue the brute force search until we find a matching e (found) or we exhaust all options (finished)
    #   We use mCx: x starts as 0 error slots, 1 error slot, 2 error slots etc...
    #   For each combination in e of a fixed number of error slots (r), there are 2 options for each error slot (1 and -1)
    while (not found) and (not finished):

        # List containing all combinations with r error slots available 
        ind_combs = list(it.combinations(indeces, r))

        # Builds the options of 1 and -1 for each error slot combination
        options = [[1, -1]]*r
        it_opt = []
        for i in it.product(*options):
            it_opt.append(i)

        # Trials all combinations of error vectors, e, at the current iteraton of r number of error slots 
        for i in range(len(ind_combs)):
            for j in range(len(it_opt)):
                e_trial = [0] * len(A)
                slots = ind_combs[i]
                vals = it_opt[j]
                for k in range(len(slots)):
                    e_trial[slots[k]] = vals[k]

                # The code for checking if A.(A_left.(b - e)) - b = -e
                e = np.mod(e_trial, q).reshape((len(e_trial), 1))
                e = GF(e)
                b_e = b - e
                Al_b_e = np.matmul(A_left, b_e) # Note that this is equal to s, if e trial was a match
                res = np.matmul(A, Al_b_e) - b
                _e = -1 * e
                match = (res == _e).all()

                # If we find a match, we can terminate early
                if match:
                    found = True
                    break

            # Early exit
            if found:
                break

        # Increment r for the next iteration, and stop the search if we finished searching all error vectors e
        r += 1
        if r > len(indeces):
            finished = True

    # We can assume we have exited with a matching, so extract s from the most recent iteration and convert it into a numpy array
    s = Al_b_e
    s = np.array(s)

    # Return the result of decrypting the ciphertext with the s we found
    return decrypt(ciphertext, s, q)


    







"""Computes standard Euclidean Norm - used in crack 3"""
"""
    - v: vector 
    - norm_aux: used in calculating meaningful distances in modulo q
"""
def norm(v, norm_aux):
    sum = 0
    for i in range(len(v)):
        sum += (norm_aux[v[i][0]])**2
    return sum**0.5






"""Crack 3 - Learning With Errors"""
"""
    - ciphertext: array of a' along with bit strings to be decrypted 
    - public_key: of the form (A, b) 
    - q: the modulo group we work in, Zq
"""
def crack3(ciphertext, public_key, q):
    # Create Galois Field object to work in modulo q
    GF = galois.GF(q)

    # Extract A, b from the public key
    A, b = public_key

    # Creating lattice basis B, and using galois to initialise it in Zq
    B = np.concatenate((A, np.zeros((1, len(A[0])), dtype=int)), axis = 0)
    B = np.concatenate((B, np.concatenate(( b.reshape((len(b), 1)), np.array([[1]])), axis = 0)), axis=1)
    B = GF(B)
    
    # Used in computing norms of vectors
    # The intuition behind this is that in modulo q: 0, q, 2q etc.. should all be considered 0
    # Therefore q-1 should be considered just as close to 0 as 1, as with q-2 and 2 etc...
    norm_aux = [0] * q
    for i in range(q):
        if abs(i-q) < i:
            norm_aux[i] = i-q
        else:
            norm_aux[i] = i

    # We search for the shortest vector using a differences sieve procedure
    # Start by creating <bag_size> number of lattice points, created from random linear combinations of the basis vectors in B
    #   Personal experimentation suggests 200 is a reasonable quantity for this coursework's n,m,q parameters
    #       - ~90% chance of finding the shortest vector in one run, duration: 1min to 1min 20s (on home laptop)
    #       - Typically won't have to repeat the procedure more than once

    # While there exists two vectors in our bag st: |v-w| < v, we will rewrite v as v-w in our bag
    #   - This method of differences is computationally efficient since we are guaranteed to obtain vector in the lattice without extra work
    #   - We also prevent the bag from growing exponentially large, slowing down the computation
    #       - Note that the vectors GRADUALLY decrease in size to prevent convergence to a local minima within our bag
    #   - LLL does not necessarily assist this algorithm since randomly generated matrices A tend to have angles of deviation in the Vector space that are large enough

    # Creating the bag
    v_bag = []
    bag_size = 200
    v_bag_norms = []
    for i in range(bag_size):
        xy = GF.Random((len(B[0]), 1))
        v = np.matmul(B, xy)
        v_bag.append(v)
        v_bag_norms.append(norm(v, norm_aux))

    # Initialise the index in our bag of vectors containing the shortest vector and its size, as well as runtime parameters (run, repeat)
    sv_index = 0
    sv_size = ((q**2) * len(B)) * 0.5 
    run = True
    repeat = True

    # Repeat the procedure until we find an assumed shortest vector with 1 or -1 y values of b
    while repeat:
        while run:

            # Run is set to be true if a new shorter vector is found
            run = False

            # Compare every pair of vectors in the bag via differences and calculate the respective norm
            for i in range(bag_size-1):
                for j in range(i+1, bag_size):
                    diff = v_bag[i] - v_bag[j]
                    diff_norm = norm(diff, norm_aux)

                    # If we find a zero vector, ignore it since we are looing for the shortest non-zero vector
                    if diff_norm == 0:
                        continue

                    # If we find a new shorter vector, update this vector in the bag
                    if diff_norm < v_bag_norms[i]:
                        v_bag[i] = diff
                        v_bag_norms[i] = diff_norm

                        # Check if this new shorter vector is shorter than our current shortest, and update the index in the bag accordingly if so
                        if diff_norm < sv_size:
                            sv_index = i
                            sv_size = diff_norm
                        
                        # The comparison procedure can continue since there has been a change in the bag of vectors
                        run = True

        # If our procedure finds the shortest vector, we can end the search
        if (v_bag[sv_index][-1][0] == 1) or (v_bag[sv_index][-1][0] == q-1):
            repeat = False
            run = False

        # ... otherwise, reset the parameters and repeat the search
        else:
            # Creating the bag
            v_bag = []
            bag_size = 200
            v_bag_norms = []
            for i in range(bag_size):
                xy = GF.Random((len(B[0]), 1))
                v = np.matmul(B, xy)
                v_bag.append(v)
                v_bag_norms.append(norm(v, norm_aux))

            # Initialise the index in our bag of vectors containing the shortest vector and its size, as well as runtime parameters (run, repeat)
            sv_index = 0
            sv_size = ((q**2) * len(B)) * 0.5 
            run = True
            repeat = True
            run = True

    # Extract the shortest vector found and rectify the error vector to have y = -1
    sv = v_bag[sv_index]
    if sv[-1][0] == 1:
        sv = -1 * sv

    # We are now very likely to have the correct shortest vector, with appropriate y value as -1
    # (... -1) means ... = Ax - b, so the shortest vector represents (-e -1)
    # -e = A.s - b
    # A.s = -e + b
    # A_left.(-e + b)  = s, the private key

    # Initialise A and b as Zq fields with galois
    A = GF(A)
    b = GF(b)

    # Compute the left inverse of A (similar to crack 1)
    At = np.transpose(A)
    At_A = np.matmul(At, A)
    At_A_inv = np.linalg.inv(At_A)
    A_left = np.matmul(At_A_inv, At)

    # Extract s as above equations dictate, and convert it into a numpy array
    s = np.matmul(A_left, sv[:-1] + b.reshape((len(b), 1)))
    s = np.array(s)

    # Return the result of decrypting the ciphertext with the s we found
    return decrypt(ciphertext, s, q)




        




