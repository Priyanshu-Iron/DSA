// Two Sum
/*
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1]
*/

// let nums = [2, 7, 11, 15];
// let target = 13;
// let sum = 0;
// let found = false; // To track if the target has been found


// Brute Force
// for(let i = 0; i < nums.length; i++){
//     for(let j = i + 1; j < nums.length; j++){
//         sum = nums[i] + nums[j];
//         if(sum === target){
//             console.log(`Array of nums i = ${nums[i]} & Array of nums j = ${nums[j]} is equal to target ${target}`);
//             console.log(`Indices: [${i}, ${j}]`);
//             found = true;
//             break; // Exit inner loop once target is found
//         }
//     }
//     if(found) break; // Exit outer loop if target is found
// }

// if(!found) {
//     console.log(`Target value ${target} not found in the given array.`);
// }

// Better Solution
// function TwoSums (nums,target){
//     let hashMap = {};
//     for(let i=0; i<nums.length; i++){
//         let a = nums[i]
//         let more = target-a;
//         if(more in hashMap){
//             return ["YES",hashMap[more],[i]]
//         }
//         hashMap[a]=i
//     }
//     return "No";
// }

// console.log(TwoSums(nums,target));


// Optimal Solution
let nums = [3,2,4];
let target = 6;
let left = 0;
let right = nums.length - 1;
let sum = 0;

nums.sort((a, b) => a - b); // Sorting the array in ascending order

while (left < right) {
    sum = nums[left] + nums[right]; // Calculate the sum of the current pair
    
    if (sum === target) {
        console.log(`Pair found at indices: ${left} and ${right}`);
        console.log(`Numbers are: ${nums[left]} and ${nums[right]}`);
        break;
    } else if (sum > target) {
        right--; // Decrease the right pointer to try smaller values
    } else {
        left++; // Increase the left pointer to try larger values
    }
}

if (left >= right) {
    console.log("No pair found that adds up to the target.");
}
