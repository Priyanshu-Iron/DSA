#include <vector>
#include <iostream>

using namespace std;

int SearchInsertPosition(vector<int> arr, int n, int x) {
    int low = 0, high = n - 1;
    int ans = n; // Default to n if x is greater than all elements

    while (low <= high) {
        int mid = (low + high) / 2;
        // If we find an element equal to or greater than x
        if (arr[mid] >= x) {
            ans = mid; // Update answer to current index
            high = mid - 1; // Look for a smaller index
        } else {
            low = mid + 1; // Look for a larger index
        }
    }

    return ans;
}   

int main() {
    int n, x;
    cout << "Enter number of elements: ";
    cin >> n;
    vector<int> arr(n);
    cout << "Enter sorted array elements: ";
    for (int i = 0; i < n; ++i) {
        cin >> arr[i];
    }
    cout << "Enter value to find insert position: ";
    cin >> x;

    int result = SearchInsertPosition(arr, n, x);
    cout << "Insert position for " << x << " is at index: " << result << endl;

    return 0;
}