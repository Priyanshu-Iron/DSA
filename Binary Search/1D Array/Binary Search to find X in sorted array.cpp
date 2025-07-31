#include <vector>
#include <iostream>
 
using namespace std;

int search(vector<int>& nums, int target) {
    int n = nums.size();
    int low = 0, high = n - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (nums[mid] == target)
            return mid;
        else if (target > nums[mid])
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}

int main() {
    int n, target;
    cout << "Enter number of elements: ";
    cin >> n;
    vector<int> nums(n);
    cout << "Enter sorted array elements: ";
    for (int i = 0; i < n; ++i) {
        cin >> nums[i];
    }
    cout << "Enter target to search: ";
    cin >> target;
    int result = search(nums, target);
    if (result != -1)
        cout << "Target found at index: " << result << endl;
    else
        cout << "Target not found in array." << endl;
    return 0;
}