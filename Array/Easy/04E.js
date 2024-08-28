// Smallest Element in an Array

let arr = [2,4,2,1,0,45,6]

function Smallest(arr){
    let smallest = arr[0]
    for(let i=1; i<arr.length; i++){
        if (arr[i]<smallest) {
            smallest = arr[i]
        }
    }
    console.log(smallest);
}
Smallest(arr);