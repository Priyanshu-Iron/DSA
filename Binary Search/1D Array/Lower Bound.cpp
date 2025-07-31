#include <vector>
#include <iostream>

using namespace std;

int lowerbound(vector<int> arr, int n, int x){
    int low = 0, high = n-1;
    int ans = n;

    while(low<=high){
        int mid = (low+high)/2;
        // maybe an answer
        if(arr[mid] >= x){
            ans = mid; // update answer
            high = mid - 1; // look for a smaller index
        } else {
            low = mid + 1; // look for a larger index
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
    cout << "Enter value to find lower bound: ";
    cin >> x;

    int result = lowerbound(arr, n, x);
    if (result < n)
        cout << "Lower bound of " << x << " is at index: " << result << endl;
    else
        cout << "Lower bound not found in the array." << endl;

    return 0;
}