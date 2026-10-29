// Local OCR only: read a PNG from stdin and return text with normalized bounds.
import Foundation
import Vision
import ImageIO

do {
    let data = FileHandle.standardInput.readDataToEndOfFile()
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "TurnText", code: 1)
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.recognitionLanguages = ["en-US"]
    try VNImageRequestHandler(cgImage: image).perform([request])
    let rows: [[String: Any]] = (request.results ?? []).compactMap { result in
        guard let text = result.topCandidates(1).first else { return nil }
        let box = result.boundingBox
        return ["text": text.string, "confidence": text.confidence,
                "x": box.minX, "y": 1 - box.maxY,
                "width": box.width, "height": box.height]
    }
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: rows))
} catch {
    FileHandle.standardError.write(Data("Local turn text recognition failed: \(error)".utf8))
    exit(1)
}
