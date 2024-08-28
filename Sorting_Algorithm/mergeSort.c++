#include <iostream>
#include <vector>

void merge(int arr[], int l, int m, int r) {
    std::vector<int> temp;
    int left = l;
    int right = m + 1;

    // Merge the two halves into temp vector
    while (left <= m && right <= r) {
        if (arr[left] <= arr[right]) {
            temp.push_back(arr[left]);
            left++;
        } else {
            temp.push_back(arr[right]);
            right++;
        }
    }

    // Copy the remaining elements of left half, if any
    while (left <= m) {
        temp.push_back(arr[left]);
        left++;
    }

    // Copy the remaining elements of right half, if any
    while (right <= r) {
        temp.push_back(arr[right]);
        right++;
    }

    // Copy the merged subarray into the original array
    for (int i = l; i <= r; i++) {
        arr[i] = temp[i - l];
    }
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2; // Avoid overflow

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}

int main() {
    int arr[] = {38, 27, 43, 3, 9, 82, 10};
    int arr_size = sizeof(arr) / sizeof(arr[0]);

    mergeSort(arr, 0, arr_size - 1);

    std::cout << "Sorted array: ";
    for (int i = 0; i < arr_size; i++) {
        std::cout << arr[i] << " ";
    }
    return 0;
}