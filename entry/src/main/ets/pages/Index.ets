import { hilog } from '@kit.PerformanceAnalysisKit';
import testNapi from 'libentry.so';
import aki from 'libentry.so'

const DOMAIN = 0x0000;

@Entry
@Component
struct Index {
  @State message: string = 'Hello World';

  build() {
    Row() {
      Column() {
        Text(this.message)
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            console.info(aki.SayHello("hello world")); // 调用.so文件中的代码接口
          })
      }
      .width('100%')
    }
    .height('100%')
  }
}
