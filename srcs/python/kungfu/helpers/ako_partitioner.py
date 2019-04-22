import sys

class AkoPartitioner:
    def __init__(self, num_partitions):
        self.num_partitions = num_partitions

    def __get_size(self, tensor):
        return tensor.shape.num_elements() * tensor.dtype.size

     # https://www8.cs.umu.se/kurser/TDBA77/VT06/algorithms/BOOK/BOOK2/NODE45.HTM
    def reconstruct_partition(self, grads_and_vars, D):
        k = self.num_partitions
        result = []
        n = len(D)
        k = k - 2
        while k >= 0:
            inner = []
            for i in range(D[n - 1][k] + 1, n + 1):
                inner.append(grads_and_vars[i])
            result.append(inner)
            n = D[n - 1][k]
            k -= 1

        inner = []
        for i in range(n + 1):
            inner.append(grads_and_vars[i])
        result.append(inner)
        result.reverse()
        return result

    def partition_positions(self, grads_and_vars):
        grads_sizes = [self.__get_size(g) for g, _v in grads_and_vars]

        k = self.num_partitions
        n = len(grads_sizes)
        # M[n][k] array of size n divided into k
        M = [[0 for i in range(k)] for j in range(n)]
        # D[n - 1][k - 1] separators
        D = [[0 for i in range(k - 1)] for j in range(n - 1)]

        M[0][0] = grads_sizes[0]
        # prefix sums
        for i in range(1, n):
            M[i][0] = M[i - 1][0] + grads_sizes[i]

        # init boundary condition
        for i in range(1, k):
            M[0][i] = grads_sizes[0]

        for i in range(1, n):
            for j in range(1, k):
                current_min = -1
                min_separator_pos = sys.maxsize
                for pos in range(i):
                    s = max(M[pos][j - 1], M[i][0] - M[pos][0])
                    if current_min < 0 or s < current_min:
                        current_min = s
                        min_separator_pos = pos
                M[i][j] = current_min
                D[i - 1][j - 1] = min_separator_pos
        return D

    def print_gradient_info(self, grads_and_vars):
        print("AkoLog> Current Model has {:d} gradients to be negotiated".format(len(grads_and_vars)))
        print("AkoLog> The size of each gradient is:")
        for grad, var in grads_and_vars:
            grad_size = grad.shape.num_elements() * grad.dtype.size
            print("AkoLog> Gradient: {:d} bytes".format(grad_size)) 

    def print_partition_info(self, num_partitions, partitions):
        print("AkoLog> Number of Ako Partitions: " + str(num_partitions))
        print("AkoLog> Length of partitions list: " + str(len(partitions)))
        print("AkoLog> The partition sizes are: ")
        for partition_id in range(len(partitions)):
            curr_partition = partitions[partition_id]
            part_size = sum(map(lambda grad_var: grad_var[0].shape.num_elements() * grad_var[0].dtype.size, curr_partition))
            print("AkoLog> Partition {:d}: {:d} bytes".format(partition_id, part_size))
