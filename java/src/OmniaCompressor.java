// Omnia SDK — Java API
// Maven: <dependency><groupId>com.omnia</groupId><artifactId>omnia-sdk</artifactId></dependency>

package com.omnia;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class OmniaCompressor {
    
    private final String pythonPath;
    
    public OmniaCompressor() { this("python3"); }
    
    public OmniaCompressor(String pythonPath) {
        this.pythonPath = pythonPath;
    }
    
    /**
     * Compress DICOM directory to .omnia file.
     */
    public CompressResult compress(String inputDir, String outputPath) 
            throws IOException, InterruptedException {
        
        Path dir = Paths.get(inputDir);
        if (!Files.exists(dir)) throw new FileNotFoundException(inputDir);
        
        long origSize = Files.walk(dir)
            .filter(p -> p.toString().endsWith(".dcm"))
            .mapToLong(p -> p.toFile().length())
            .sum();
        
        ProcessBuilder pb = new ProcessBuilder(
            pythonPath, "-m", "omnia", "compress", inputDir, outputPath);
        pb.inheritIO();
        Process proc = pb.start();
        int exitCode = proc.waitFor();
        
        long compSize = Files.exists(Paths.get(outputPath)) 
            ? Files.size(Paths.get(outputPath)) : 0;
        
        return new CompressResult(exitCode == 0, origSize, compSize,
            compSize > 0 ? (double)origSize / compSize : 0.0);
    }
    
    /**
     * Decompress .omnia file to DICOM directory.
     */
    public int decompress(String inputPath, String outputDir) 
            throws IOException, InterruptedException {
        
        if (!Files.exists(Paths.get(inputPath)))
            throw new FileNotFoundException(inputPath);
        
        Files.createDirectories(Paths.get(outputDir));
        
        ProcessBuilder pb = new ProcessBuilder(
            pythonPath, "-m", "omnia", "decompress", inputPath, outputDir);
        pb.inheritIO();
        Process proc = pb.start();
        proc.waitFor();
        
        return (int)Files.walk(Paths.get(outputDir))
            .filter(p -> p.toString().endsWith(".dcm"))
            .count();
    }
    
    public static class CompressResult {
        public final boolean success;
        public final long originalSize;
        public final long compressedSize;
        public final double ratio;
        
        public CompressResult(boolean s, long o, long c, double r) {
            success = s; originalSize = o; compressedSize = c; ratio = r;
        }
    }
}
