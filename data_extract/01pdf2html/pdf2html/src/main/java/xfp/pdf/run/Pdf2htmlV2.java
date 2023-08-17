package xfp.pdf.run;

import org.apache.pdfbox.pdmodel.PDDocument;
import xfp.pdf.arrange.MarkPdf;
import xfp.pdf.core.PdfParser;
import xfp.pdf.pojo.ContentPojo;
import xfp.pdf.tools.FileTool;

import java.io.File;
import java.io.IOException;
// import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class Pdf2htmlV2 {

    public static void parsePdf(File f){
        PDDocument pdd = null;
            try {
                pdd = PDDocument.load(f);
                ContentPojo contentPojo = PdfParser.parsingUnTaggedPdfWithTableDetection(pdd);
                MarkPdf.markTitleSep(contentPojo);
                // System.out.println(Path.outputAllHtmlPath);
                // System.out.println(f.getAbsolutePath() + " 解析完成");
                FileTool.saveHTML(Path.outputAllHtmlPath, contentPojo, f.getAbsolutePath());
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    pdd.close();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
    }

    public static void main(String[] args) throws IOException {

        File file = new File(Path.inputAllPdfPath);
        File[] files = file.listFiles();
        // Arrays.sort(files);
        // int part = 1;
        // File[] partFiles = Arrays.copyOfRange(files, 1000*part, 1000*(part+1));
        // for (File f : partFiles) {
        ExecutorService fixedThreadPool = Executors.newFixedThreadPool(15);
        for (File f : files) {
            fixedThreadPool.execute(()->{
                parsePdf(f);
            });
            // PDDocument pdd = null;
            // try {
            //     pdd = PDDocument.load(f);
            //     ContentPojo contentPojo = PdfParser.parsingUnTaggedPdfWithTableDetection(pdd);
            //     MarkPdf.markTitleSep(contentPojo);
            //     // System.out.println(Path.outputAllHtmlPath);
            //     // System.out.println(f.getAbsolutePath() + " 解析完成");
            //     FileTool.saveHTML(Path.outputAllHtmlPath, contentPojo, f.getAbsolutePath());
            // } catch (IOException e) {
            //     e.printStackTrace();
            // } finally {
            //     try {
            //         pdd.close();
            //     } catch (IOException e) {
            //         throw new RuntimeException(e);
            //     }
            // }
        }
    }
}