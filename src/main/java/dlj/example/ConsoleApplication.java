package dlj.example;


import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.function.Supplier;

import javax.annotation.Resource;
import javax.imageio.ImageIO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

@SpringBootApplication
public class ConsoleApplication implements CommandLineRunner {

	private static Logger LOG = LoggerFactory.getLogger(ConsoleApplication.class);

	@Resource
	private Supplier<Predictor<BufferedImage, DetectedObjects>> predictorProvider;

	public static void main(final String[] args) {
		SpringApplication.run(ConsoleApplication.class, args);
	}

	@Override
	public void run(final String... args) throws Exception {
		DetectedObjects results1 = predict1("/puppy-in-white-and-red-polka.jpg");
		results1.items().stream().filter(r -> r.getProbability() > 0.1d).map(p -> p.toString()).forEach(LOG::info);


		DetectedObjects results2 = predict2("/puppy-in-white-and-red-polka.jpg");
		results2.items().stream().filter(r -> r.getProbability() > 0.1d).map(p -> p.toString()).forEach(LOG::info);
	}

	private DetectedObjects predict1(final String image) throws TranslateException, IOException {
		try (Predictor<BufferedImage, DetectedObjects> predictor = predictorProvider.get()) {
			DetectedObjects results = predictor.predict(loadImage(image));
			return results;
		}
	}

	private DetectedObjects predict2(final String image) throws Exception {
		try (Predictor<BufferedImage, DetectedObjects> predictor = predictor()) {
			DetectedObjects results = predictor.predict(loadImage(image));
			return results;
		}
	}

	private BufferedImage loadImage(final String name) throws IOException {
		return ImageIO.read(this.getClass().getResourceAsStream(name));
	}

	private Predictor<BufferedImage, DetectedObjects> predictor() throws Exception {
		Criteria<BufferedImage, DetectedObjects> criteria = Criteria.builder()
				.optApplication(Application.CV.OBJECT_DETECTION).setTypes(BufferedImage.class, DetectedObjects.class)
				.optFilter("size", "512").optFilter("backbone", "mobilenet1.0").build();

		ZooModel<BufferedImage, DetectedObjects> model = ModelZoo.loadModel(criteria);
		Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor();
		return predictor;

	}
}
