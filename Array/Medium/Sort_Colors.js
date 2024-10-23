/*
75. Sort Colors

Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.

Example 1:
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]

Example 2:
Input: nums = [2,0,1]
Output: [0,1,2]
*/

// let nums = [2,0,2,1,1,0]

// Better Solution
/*
let count0 = 0;
let count1 = 0;
let count2 = 0;

for(let i=0; i<nums.length; i++){
    if(nums[i]==0){count0++}
    else if(nums[i]==1){count1++}
    else{count2++}
}

for(let i=0; i<count0; i++){
    nums[i]=0
}
for(let i=count0; i<count0+count1; i++){
    nums[i]=1
}
for(let i=count0+count1; i<nums.length; i++){
    nums[i]=2
}

for(let i=0; i<nums.length; i++){
    console.log(nums[i]);
}
*/
// TC = O(N)+O(N) and SC = O(1) 

// Optimal Solution
// Dutch National Flag Algorithm

let nums = [2,0,2,1,1,0]

function SortColors (nums){
    let low = 0;
    let mid = 0
    let high = nums.length-1

    while(mid<=high){
        if(nums[mid] === 0){
            let temp = nums[mid]
            nums[mid] = nums[low]
            nums[low] = temp;
            low++;
            mid++
        }
        else if(nums[mid] === 1){
            mid++
        }
        else {
            let temp = nums[mid];
            nums[mid] = nums[high];
            nums[high] = temp;
            high--;
        }
    }
    return nums
}

console.log(SortColors(nums));

// TC = O(N) and and SC = O(N)