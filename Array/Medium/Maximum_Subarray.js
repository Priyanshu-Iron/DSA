// 53. Maximum Subarray

/*
Given an integer array nums, find the subarray with the largest sum, and return its sum.

Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
*/

// node Maximum_Subarray.js

let nums = [-2,1,-3,4,-1,2,1,-5,4]

// Brute Force
/*
function MaximumSubarray (nums){
    let maxi = Number.MIN_SAFE_INTEGER
    for(let i=0; i<nums.length; i++){
        for(let j=i; j<nums.length; j++){
            let sum =0;
            for(let k=i; k<=j; k++){
                sum += nums[k];
            }
            maxi = Math.max(maxi,sum)
        }
    }
    return maxi;
}

console.log(MaximumSubarray(nums))
*/

// Optimal solution
/*
function MaximumSubarrayBetter (nums) {
    let maxi = Number.MIN_SAFE_INTEGER
    for(let i=0; i<nums.length; i++){
        let sum = 0;
        for(let j=i; j<nums.length; j++){
            sum += nums[j]
            maxi = Math.max(maxi,sum)
        }
    }
    return maxi
}

console.log(MaximumSubarrayBetter(nums))
*/

function MaximumSubarrayOptimal (nums) {
    let sum = 0
    let maxi = nums[0]

    for(let i=0; i<nums.length; i++){
        sum += nums[i]

        maxi = Math.max(maxi,sum);

        if(sum<0){
            sum=0
        }
    }
    return maxi;
}

console.log(MaximumSubarrayOptimal(nums))