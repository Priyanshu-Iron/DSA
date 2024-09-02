let array = [1,2,3,4,5];

// Rotate Array Left
function RotateArrayLeft (array){
    let temp = array[0]
    let n = array.length;
    for(let i = 1; i<n; i++){
        array[i-1] = array[i];
    }
    array[n-1] = temp;
    console.log(array);
}
// RotateArrayLeft(array);

// Rotate Array Right
function RotateArrayRight (array){
    let n = array.length
    let temp = array[n-1]
    for(let i=n-1; i>0; i--){
        array[i]=array[i-1]
    }
    array[0] = temp;
    console.log(array);
}

RotateArrayRight(array);