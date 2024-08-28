// Largest Element in an Array

let arr = [2,4,5,2,5,6]

function largest(arr){
    let largest = arr[0]
    for(let i=1; i<arr.length; i++){
        if (arr[i]>largest) {
            largest=arr[i]
        }
    }
    console.log(largest);
}
largest(arr)