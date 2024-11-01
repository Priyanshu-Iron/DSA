/*
35. Search Insert Position

Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
You must write an algorithm with O(log n) runtime complexity.

Example 1:
Input: nums = [1,3,5,6], target = 5
Output: 2

Example 2:
Input: nums = [1,3,5,6], target = 2
Output: 1
*/
// node SearchInsertPosition.js

let  nums = [1,3,5,6]
let target = 7

/*
function  SearchInsertPosition (nums,target) {
    for(let i=0; i<nums.length; i++){
        if(target <= nums[i]){
            return i
        }
    }
    return nums.length;
}

console.log(SearchInsertPosition(nums,target));
*/


// Using O(log N) TC
// Binary Search
function SearchInsertPosition(nums,target){
    let left = 0;
    let right = nums.length - 1

    while(left <= right){
        const mid = Math.floor((left+right)/2)
        if(nums[mid] === target){
            return mid;
        } else if(nums[mid] < target){
            left = mid+1;
        }else{
            right = mid-1
        }
    }
    return left
}

console.log(SearchInsertPosition(nums,target));