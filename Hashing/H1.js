const array = [1,2,1,2,4,1]
const number = 1

function WhyHashing () {
    let count = 0;
    for(let i=0; i<array.length; i++){
        if(array[i]==number){
            count = count + 1
        }
    }
    return count
}

console.log(WhyHashing());