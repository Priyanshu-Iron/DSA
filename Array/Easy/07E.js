// Remove Duplicate from Array

let array = [0,0,1,1,1,2,2,3,3,4]

function RemoveDuplicate(array){
    let i = 0;
    for (let j = 0; j < array.length; j++) {
        if (array[i] != array[j]) {
            array[i+1] = array[j];
            i++;
        };
    };
    console.log(i+1);

    // For Seeing New Array
    // console.log(array.slice(0,i+1));
    
}

RemoveDuplicate(array)

// For Seeing New Array
// function newArray(){RemoveDuplicate(array);}
// newArray(array)