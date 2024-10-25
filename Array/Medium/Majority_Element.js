// 169. Majority Element

/*
Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

Example 1:
Input: nums = [3,2,3]
Output: 3

Example 2:
Input: nums = [2,2,1,1,1,2,2]
Output: 2
*/

let nums = [2,2,1,1,1,2,2]

// Brute Force Solution
/*
function MajorityElement (nums){
    for(let i=0; i<nums.length; i++){

        let count = 0;
    
        for(let j=0; j<nums.length; j++){
            if(nums[i] == nums[j]){
                count++;
            }
        }
    
        if(count>nums.length/2){
            return nums[i]
        }
    }
}

console.log(MajorityElement(nums));
*/

// node Majority_Element.js

// Better Solution

function MajorityElement(nums) {
    const map = new Map();

    for (let i = 0; i < nums.length; i++) {
        if (map.has(nums[i])) {
            map.set(nums[i], map.get(nums[i]) + 1);
        } else {
            map.set(nums[i], 1);
        }
    }

    const majorityCount = Math.floor(nums.length / 2);
    for (const [num, count] of map) {
        if (count > majorityCount) {
            return num;
        }
    }

    return -1;
}

console.log(MajorityElement(nums));