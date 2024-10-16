// 896. Monotonic Array
/*
A monotonic array is an array that is either entirely non-increasing or entirely non-decreasing. 
In simpler terms:
Monotonically increasing: Each element is greater than or equal to the preceding element.
Monotonically decreasing: Each element is less than or equal to the preceding element

[1, 2, 2, 3]is monotonically increasing.
[6, 5, 4, 4]is monotonically decreasing.
[1, 3, 2]is not monotonic because it increases and then decreases
*/

let nums = [6,5,4,4]

function MonotonicArray (nums){

    let increasing = true
    let decreasing = true

    for(let i=0; i<nums.length-1; i++){
        if (nums[i] > nums[i+1]){
            increasing = false;
        }
        if (nums[i] < nums[i + 1]) {
            decreasing = false;
        }
        if (!increasing && !decreasing) {
            return false;
        }
    }
    return increasing || decreasing;
}

console.log(MonotonicArray(nums));