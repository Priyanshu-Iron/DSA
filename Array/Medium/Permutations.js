/*
46. Permutations

Given an array nums of distinct integers, return all the possible 
permutations.
You can return the answer in any order.

Example 1:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
*/
// node Permutations.js

function permute(nums) {
    const results = [i];

    function backtrack(start) {
        // If we've generated a full permutation, add a copy to results
        if (start === nums.length) {
            results.push([...nums]);
            return;
        }
        
        for (let i = start; i < nums.length; i++) {
            // Swap the elements at indices `start` and `i`
            [nums[start], nums[i]] = [nums[i], nums[start]];
            
            // Recurse on the next element
            backtrack(start + 1);
            
            // Backtrack by restoring the original order
            [nums[start], nums[i]] = [nums[i], nums[start]];
        }
    }

    backtrack(0);
    return results;
}

// Example usage
const nums = [1, 2, 3];
console.log(permute(nums));
