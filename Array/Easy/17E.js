// 561. Array Partition

/*
Input: nums = [1,4,3,2]
Output: 4
Explanation: All possible pairings (ignoring the ordering of elements) are:
1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3
2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3
3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4
So the maximum possible sum is 4.
*/

/*
STEPS TO SOLVE :-
1. Array Sorting
2. Iterating the Array with 2 element and suming them
*/

let nums = [6,2,6,5,1,2]

function ArrayPartition (nums){
    nums.sort((a, b) => a - b);

    let maxSum = 0 

    for(let i=0; i< nums.length; i+=2){
        maxSum += nums[i];
    }

    return maxSum
}

console.log(ArrayPartition(nums));