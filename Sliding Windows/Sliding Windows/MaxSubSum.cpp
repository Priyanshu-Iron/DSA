#include <iostream>
#include <vector>
#include <algorithm> // for std::max

int MaxSubArrSum(const std::vector<int>& array, int k) {
    int i = 0, j = 0;
    int sum = 0;
    int maxSum = array[0];

    while (j < array.size()) {
        sum += array[j];

        if (j - i + 1 < k) {
            j++;
        } else if (j - i + 1 == k) {
            maxSum = std::max(maxSum, sum);
            sum -= array[i];
            i++;
            j++;
        }
    }

    return maxSum;
}

int main() {
    std::vector<int> array = {1,5,4,2,9,9,9};
    int k = 3;
    std::cout << "Max sum of subarray of size " << k << ": " << MaxSubArrSum(array, k) << std::endl;
    return 0;
}

