// 3131. Find the Integer Added to Array I
/*
You are given two arrays of equal length, nums1 and nums2.
Each element in nums1 has been increased (or decreased in the case of negative) by an integer, represented by the variable x.
As a result, nums1 becomes equal to nums2. Two arrays are considered equal when they contain the same integers with the same frequencies.
Return the integer x.

Example 1:
Input: nums1 = [2,6,4], nums2 = [9,7,5]
Output: 3
Explanation:
The integer added to each element of nums1 is 3.
*/
// node Integer_Added.js

let nums1 = [2,6,4]
let nums2 = [9,7,5]

function IntegerAdded (nums1,nums2) {
    nums1.sort((a, b) => a - b)
    nums2.sort((a, b) => a - b)
    
    return nums2[0]-nums1[0]
}

console.log(IntegerAdded(nums1,nums2));