#include<stdio.h>

int fibonacci_n(int n){
    if (n <= 1){
        return 1;
    }
    return fibonacci_n(n - 1) + fibonacci_n(n - 2);
}

int main() {
    int n;
    printf("Introdu numarul n pentru a calcula Fibonacci: ");
    scanf("%d", &n);
    printf("Al n-lea numar fibonacci este: %d\n", fibonacci_n(n));
    return 0;
}