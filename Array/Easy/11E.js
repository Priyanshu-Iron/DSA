// Linear Search

let nums = [2,3,4,5,6,3]
let target = 4

function Search (nums,target) {
    for(let i=0; i<nums.length; i++){
        if(nums[i] === target){
            return i;
        }
    }
    return -1;
}

console.log(Search(nums,target));