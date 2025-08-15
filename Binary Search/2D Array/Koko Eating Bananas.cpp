#include <stdio.h>
#include <vector>
#include <limits.h>
#include <iostream>

using namespace std;

int findMax(vector <int>&piles){
    int maxi = INT_MIN;
    int n = piles.size();
    // Find the Maximum:
    for(int i=0; i<n; i++){
        maxi = max(maxi, piles[i]);
    }
    return maxi;
}

int calculateTotalHours(vector<int>&piles, int hourly){
    int totalH = 0;
    int n = piles.size();
    // Find Total Hours:
    for(int i=0; i<n; i++){
        totalH += ceil((double)(piles[i])/(double)(hourly));
    }
    return totalH;
}

// LINEAR SEARCH APPROACH
int minEatingSpeed(vector<int>& piles, int h) {
    // Find the Maximum Number:
    int maxi = findMax(piles);
    // Find the minimum value of k:
    for(int i=1; i<=maxi; i++){
        int reqTime = calculateTotalHours(piles, i);
        if(reqTime<=h){
            return i;
        }
    }
    return maxi;
}

// BINARY SEARCH APPROACH
int minimumRateToEatBananas(vector<int> piles, int h) {
    int low = 1, high = findMax(piles);

    //apply binary search:
    while (low <= high) {
        int mid = (low + high) / 2;
        int totalH = calculateTotalHours(piles, mid);
        if (totalH <= h) {
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }
    return low;
}

int main(){
    vector <int> piles = {7, 15, 6, 3};
    int h = 8;
    int ans = minEatingSpeed(piles, h);
    int ans2 = minimumRateToEatBananas(piles, h);
    cout << "Minimum eating speed using linear search: " << ans << endl;
    cout << "Minimum eating speed using binary search: " << ans2 << endl;
    return 0;
}