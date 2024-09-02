// Check Array is Sorted

let nums = [2, 3, 4];
var isSorted = function(nums) {
    let ascending = true;
    let descending = true;
    
    for(let i = 1; i < nums.length; i++) {
        if(nums[i] < nums[i-1]) {
            ascending = false;
        }
        if(nums[i] > nums[i-1]) {
            descending = false;
        }
    }
    return ascending || descending;
};
console.log(isSorted(nums)); 