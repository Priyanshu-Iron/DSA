// node SumoffirstNNaturalNumbers.js

/*
function Sum (n) {
    if (n === 0) return 0;
    return n + Sum(n - 1);
}

function main () {
    let n = 4
    console.log(Sum(n));
}

main()
*/

for (let index = 1; index <= 4; index++) {
    let sum = index * (index + 1) / 2;
    if (index === 4) {
        console.log(sum);
    }
}