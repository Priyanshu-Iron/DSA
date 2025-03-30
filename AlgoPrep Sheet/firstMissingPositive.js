const nums = [2, -3, 4, 1, 1, 7];

function NaiveApproachfirstMissingPositive(nums) {
    nums.sort((a, b) => a - b);
    let missingPositive = 1;
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] === missingPositive) {
            missingPositive++;
        }
        else if (nums[i] > missingPositive) {
            break;
        }
    }
    return missingPositive;
}

console.log(NaiveApproachfirstMissingPositive(nums));


// JavaScript program to find the first missing positive number 
// using visited array

function BetterApproachmissingNumber(nums) {
    let n = nums.length;

    // To mark the occurrence of elements
    let vis = new Array(n).fill(false);
    for (let i = 0; i < n; i++) {

        // if element is in range from 1 to n
        // then mark it as visited
        if (nums[i] > 0 && nums[i] <= n)
            vis[nums[i] - 1] = true;
    }

    // Find the first element which is unvisited
    // in the original numsay
    for (let i = 1; i <= n; i++) {
        if (!vis[i - 1]) {
            return i;
        }
    }

    // if all elements from 1 to n are visited
    // then n+1 will be first positive missing number
    return n + 1;
}

console.log(BetterApproachmissingNumber(nums));


// JavaScript program to find the first missing positive number 
// using cycle sort

function OptimalmissingNumber(nums) {
    let n = nums.length;
    for (let i = 0; i < n; i++) {

        // if nums[i] is within the range 1 to n and nums[i] is
        // not placed at (nums[i]-1)th index in nums
        while (nums[i] >= 1 && nums[i] <= n 
               && nums[i] !== nums[nums[i] - 1]) {

            // then swap nums[i] and nums[nums[i]-1] to place 
            // nums[i] to its corresponding index
            let temp = nums[i];
            nums[i] = nums[nums[i] - 1];
            nums[temp - 1] = temp;
        }
    }

    // If any number is not at its corresponding index 
    // it is then missing,
    for (let i = 1; i <= n; i++) {
        if (i !== nums[i - 1]) {
            return i;
        }
    }

    // If all number from 1 to n are present 
    // then n+1 is smallest missing number
    return n + 1;
}

console.log(OptimalmissingNumber(nums));