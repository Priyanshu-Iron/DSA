/*
128. Longest Consecutive Sequence

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

Example 1:
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
*/

let nums = [100,4,200,1,3,2]

/*
function linearSearch (nums, arr) {
    for(let i=0; i<nums.length; i++){
        if(nums[i] === arr){
            return true;
        }
    }
    return false
}

function LongestConsecutiveSequence (nums) {
let longest = 1;
for(let i=0; i<nums.length; i++){
    let x = nums[i]
    let count = 1
    while( linearSearch (nums,x+1) == true ){
        x = x+1
        count += 1
    }
    longest = Math.max(longest,count)
}
return longest
}

console.log(LongestConsecutiveSequence(nums));
*/

// Optimal Solution
function LongestConsecutiveSequence(nums) {
    let numSet = new Set(nums); // Create a set to allow O(1) lookups
    let longest = 0;

    for (let num of numSet) {
        // Only start counting for the smallest number in the sequence
        if (!numSet.has(num - 1)) {
            let currentNum = num;
            let count = 1;

            while (numSet.has(currentNum + 1)) {
                currentNum += 1;
                count += 1;
            }

            longest = Math.max(longest, count);
        }
    }

    return longest;
}

console.log(LongestConsecutiveSequence(nums)); // Output: 4