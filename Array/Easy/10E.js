// Move all Zeroes to the End

let nums = [0,2,3,1,0,6,7]

// Brute Force Method
const moveZeroes = function(nums) {
    let temp = [];
    let n = nums.length;

    // Push all non-zero elements to temp
    for(let i=0; i<n; i++){
        if (nums[i] != 0) {
            temp.push(nums[i]);
        }
    }

    // Copy all non-zero elements from temp back to nums
    let nz = temp.length
    for(let i=0; i<nz; i++){
        nums[i] = temp[i];
    }

    // Fill the remaining elements with zeroes
    for(let i=nz; i<n; i++){
        nums[i]=0;
    }

    return nums;
}

// console.log(moveZeroes(nums));

// Optimal Solution
function moveZeroesOS(nums){
    let n = nums.length;
    let j = -1;

    // Find the first zero
    for(let i=0; i<n; i++){
        if(nums[i] == 0){
            j=i;
            break;
        }
    }

    // If no zero is found, return the array as is
    if(j == -1) return nums;

    // Iterate and swap non-zero elements with zeroes
    for(let i=j+1; i<n; i++){
        if(nums[i] != 0){
            // Swap elements
            [nums[i],nums[j]] = [nums[j],nums[i]];
            j++;
        }
    }

    return nums;
}

console.log(moveZeroesOS(nums));