import java.util.ArrayList;
import java.util.Scanner;

    public class Main {
    public static void main(String[] args) {
      Scanner sc = new Scanner(System.in);
      int n = sc.nextInt();
      int m = sc.nextInt();
      ArrayList<Integer> arr1 = new ArrayList<>();
      ArrayList<Integer> arr2 = new ArrayList<>();
      ArrayList<Integer> arr = new ArrayList<>();
      for(int i = 0;i<n;i++) {//获取
       arr1.add(sc.nextInt());
      }
      for(int i = 0;i<n;i++) {
       arr2.add(sc.nextInt());
      }

      for(int i = 0;i<n;i++) {//找到每个的最大
       find(arr,arr1,arr2,m);
      }
      for(int i = 0;i<n;i++) {
       System.out.print(arr.get(i)+" ");
      }
      }
    public static void find(ArrayList<Integer> arr,ArrayList<Integer> arr1,ArrayList<Integer> arr2,int m) {
    int num = 0;
    int k = 0;int t = 0;//记录下标
    for(int i = 0;i<arr1.size();i++) {
       for(int j = 0;j<arr2.size();j++) {
          if((arr1.get(i)+arr2.get(j))%m>num) {//大于预定的数
           num = (arr1.get(i)+arr2.get(j))%m;
           k = i;
           t = j;
        }
     }
     
    }
    arr1.remove(k);
    arr2.remove(t);
    arr.add(num);
    }

    }