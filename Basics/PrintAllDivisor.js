// node PrintAllDivisor.js

num = 28

function PerfectNumber () {
    let sum = 0 
    for(let i=1; i<num; i++){
        if(num%i === 0){
            sum = sum+i
        }
    }
    return sum === num
}

console.log(PerfectNumber(num));