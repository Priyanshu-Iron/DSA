// 27. Remove Element

let nums = [0,1,2,2,3,0,4,2]
let val = 2
// let count = 0

// for(let i=0; i<nums.length; i++){
//     for(let j=1; j<nums.length; j++){
//         if(nums[i]==val){
//             nums.pop()
//         }
//         count++
//     }
// }

// console.log(nums);


// let newNums = []
// for(let i=0; i<nums.length; i++){
//     if(nums[i]!=val){
//         newNums.push(nums[i])
//     }
// }

// console.log(newNums);


function removeElement(nums,val){
    let k=0;
    for(let i=0; i<nums.length; i++){
        if(nums[i]!=val){
            nums[k] = nums[i]
            k++
        }
    }

    k=nums.slice(0,k)
    return k
}

console.log(removeElement(nums,val));
