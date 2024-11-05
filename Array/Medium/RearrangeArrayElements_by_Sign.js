/*
2149. Rearrange Array Elements by Sign

You are given a 0-indexed integer array nums of even length consisting of an equal number of positive and negative integers.
You should return the array of nums such that the the array follows the given conditions:
Every consecutive pair of integers have opposite signs.
For all integers with the same sign, the order in which they were present in nums is preserved.
The rearranged array begins with a positive integer.
Return the modified array after rearranging the elements to satisfy the aforementioned conditions.

Example 1:
Input: nums = [3,1,-2,-5,2,-4]
Output: [3,-2,1,-5,2,-4]
Explanation:
The positive integers in nums are [3,1,2]. The negative integers are [-2,-5,-4].
The only possible way to rearrange them such that they satisfy all conditions is [3,-2,1,-5,2,-4].
Other ways such as [1,-2,2,-5,3,-4], [3,1,2,-2,-5,-4], [-2,3,-5,1,-4,2] are incorrect because they do not satisfy one or more conditions.
*/
// node RearrangeArrayElements_by_Sign.js

let nums = [3,1,-2,-5,2,-4]

// BruteForce Method
/*
function RearrangeArrayElementsBySign (nums){

    let positive = []
    let negaitive = []

    for(let i=0; i<nums.length; i++){
        if(nums[i]>0){
            positive.push(nums[i])
        }else{
            negaitive.push(nums[i])
        }
    }

    for(let i=0; i<nums.length/2; i++){
        nums[2*i] = positive[i]
        nums[2*i+1] = negaitive[i]
    }

    return nums;
}

console.log(RearrangeArrayElementsBySign(nums));
*/

function RearrangeArrayElementsBySign (nums){
    let newArray = []
    let positiveIndex = 0
    let negaitiveIndex = 1

    for(let i=0; i<nums.length; i++){
        if(nums[i]<0){
            newArray[negaitiveIndex] = nums[i];
            negaitiveIndex +=2
        }else{
            newArray[positiveIndex] = nums[i];
            positiveIndex +=2
        }
    }
    return newArray
}

console.log(RearrangeArrayElementsBySign(nums));