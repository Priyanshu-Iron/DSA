// 560. Subarray Sum Equals K
/*
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
A subarray is a contiguous non-empty sequence of elements within an array.

Example 1:
Input: nums = [1,1,1], k = 2
Output: 2
*/

// node Longest_SubArray.js

let nums = [1, 2, 1, 2, 1]
let k = 3

/*
function LongestSubarray (nums,k){
    let long = 0;
    for(let i=0; i<nums.length; i++){
        let sum = 0
        for(let j=i; j<nums.length; j++){
            sum += nums[j]
            if(sum==k){
                long = Math.max(long,j-i+1)
            }
        }
    }
    return long
}

console.log(LongestSubarray(nums,k));
*/

// Better Solution
function getLongestSubarray(nums, k) {
    let preSumMap = new Map();
    let sum = 0;
    let maxLen = 0;

    for (let i = 0; i < nums.length; i++) {
        sum += nums[i];

        // Check if sum from start to current index equals k
        if (sum === k) {
            maxLen = i + 1;
        }

        // Calculate remaining sum to reach k
        let rem = sum - k;

        // Check if we have seen a prefix sum that would allow a subarray of sum k
        if (preSumMap.has(rem)) {
            let len = i - preSumMap.get(rem);
            maxLen = Math.max(maxLen, len);
        }

        // Record the first occurrence of each prefix sum to maximize subarray length
        if (!preSumMap.has(sum)) {
            preSumMap.set(sum, i);
        }
    }
    return maxLen;
}

console.log(getLongestSubarray(nums,k));  // Output should be 4


/*
function subarraySum(nums, k) {
    let preSumMap = new Map();
    preSumMap.set(0, 1); // Initialize with sum 0 to count subarrays starting from the beginning
    let sum = 0;
    let count = 0;

    for (let i = 0; i < nums.length; i++) {
        sum += nums[i];

        // Check if there is a prefix subarray we can subtract to get sum k
        if (preSumMap.has(sum - k)) {
            count += preSumMap.get(sum - k);
        }

        // Update the count of current sum in the map
        preSumMap.set(sum, (preSumMap.get(sum) || 0) + 1);
    }

    return count;
}

console.log(subarraySum([1, 1, 1], 2));  // Output: 2
*/