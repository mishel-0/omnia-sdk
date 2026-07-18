// Omnia SDK — C# API
// Install: dotnet add package OmniaSDK

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace Omnia
{
    public class SliceInfo
    {
        public int Rows { get; set; }
        public int Cols { get; set; }
        public string PatientId { get; set; }
        public string StudyUid { get; set; }
    }

    public class CompressResult
    {
        public bool Success { get; set; }
        public long OriginalSize { get; set; }
        public long CompressedSize { get; set; }
        public double Ratio { get; set; }
    }

    public class OmniaCompressor
    {
        private readonly string _pythonPath;

        public OmniaCompressor(string pythonPath = "python3")
        {
            _pythonPath = pythonPath;
        }

        /// <summary>
        /// Compress DICOM directory to .omnia file.
        /// </summary>
        public CompressResult Compress(string inputDir, string outputPath)
        {
            if (!Directory.Exists(inputDir))
                throw new DirectoryNotFoundException(inputDir);

            var origSize = 0L;
            foreach (var f in Directory.GetFiles(inputDir, "*.dcm"))
                origSize += new FileInfo(f).Length;

            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = _pythonPath,
                Arguments = $"-m omnia compress \"{inputDir}\" \"{outputPath}\"",
                RedirectStandardOutput = true,
                UseShellExecute = false,
            };
            using var proc = System.Diagnostics.Process.Start(psi);
            proc.WaitForExit();

            var compSize = File.Exists(outputPath) ? new FileInfo(outputPath).Length : 0;

            return new CompressResult
            {
                Success = proc.ExitCode == 0,
                OriginalSize = origSize,
                CompressedSize = compSize,
                Ratio = compSize > 0 ? (double)origSize / compSize : 0,
            };
        }

        /// <summary>
        /// Decompress .omnia file to DICOM directory.
        /// </summary>
        public int Decompress(string inputPath, string outputDir)
        {
            if (!File.Exists(inputPath))
                throw new FileNotFoundException(inputPath);

            Directory.CreateDirectory(outputDir);

            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = _pythonPath,
                Arguments = $"-m omnia decompress \"{inputPath}\" \"{outputDir}\"",
                UseShellExecute = false,
            };
            using var proc = System.Diagnostics.Process.Start(psi);
            proc.WaitForExit();

            return Directory.GetFiles(outputDir, "*.dcm").Length;
        }
    }
}
