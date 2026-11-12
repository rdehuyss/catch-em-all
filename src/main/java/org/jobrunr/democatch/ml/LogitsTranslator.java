package org.jobrunr.democatch.ml;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDList;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/**
 * Feeds raw pixels to the frozen backbone and returns its raw output vector
 * (ImageNet logits used as features), without any zoo post-processing.
 */
public class LogitsTranslator implements Translator<Image, float[]> {

    private final int size;

    public LogitsTranslator(int size) {
        this.size = size;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        return new ai.djl.translate.Pipeline()
                .add(new Resize(size, size))
                .add(new ToTensor())
                .transform(new NDList(centerSquare(input).toNDArray(ctx.getNDManager(), Image.Flag.COLOR)));
    }

    /**
     * Webcam frames are wide while training images are square center-crops; a plain
     * resize would stretch the frame and shift the input distribution.
     */
    public static Image centerSquare(Image image) {
        int minSide = Math.min(image.getWidth(), image.getHeight());
        int x = (image.getWidth() - minSide) / 2;
        int y = (image.getHeight() - minSide) / 2;
        return image.getSubImage(x, y, minSide, minSide);
    }

    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        return list.get(0).toFloatArray();
    }
}
