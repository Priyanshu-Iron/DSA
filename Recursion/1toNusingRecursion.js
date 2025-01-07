// node 1toNusingRecursion.js

// 1toNusingRecursion
function OneToN (i,n) {
    if(i>n) return;
    console.log(i);
    OneToN(i+1,n)
}

function main () {
    let n = 4
    OneToN(1,n)
}

main()

// Nto1usingRecursion
/*
function OneToN (i,n) {
    if(i<1) return;
    console.log(i);
    OneToN(i-1,n)
}

function main () {
    let n = 4
    OneToN(n,n)
}
main()
*/