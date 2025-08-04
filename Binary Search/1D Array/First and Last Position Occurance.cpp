// Include vector header

#include <vector>
using namespace std;

class Solution {
public:
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

    int upperbound(vector<int> arr, int n, int x){
    int low = 0, high = n-1;
    int ans = n;

    while(low<=high){
        int mid = (low+high)/2;
        // maybe an answer
        if(arr[mid] > x){
            ans = mid; // update answer
            high = mid - 1; // look for a smaller index
        } else {
            low = mid + 1; // look for a larger index
        }
    }

    return ans;
    }

    vector<int> searchRange(vector<int>& nums, int target) {
        int n = nums.size();
        int lb = lowerbound(nums, n, target);
        if(lb == n || nums[lb] != target) return vector<int>{-1, -1};
        int ub = upperbound(nums, n, target);
        return vector<int>{lb, ub - 1};
    }
};